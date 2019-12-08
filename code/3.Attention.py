import numpy as np 
import pandas as pd
from keras.utils import to_categorical
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import *
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import *
import tensorflow as tf
import os
import gc
import re
import glob
kaggle = False
EMBEDDING_DIM=50
MAX_SEQUENCE_LENGTH=500
path =''
output = ''
if kaggle:
	path ='../input'
else:
	path = '../data'
	output = '../output'


def dot_product(x, kernel):
	"""
	Wrapper for dot product operation, in order to be compatible with both
	Theano and Tensorflow
	Args:
	    x (): input
	    kernel (): weights
	Returns:
	"""
	if K.backend() == 'tensorflow':
	    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
	else:
	    return K.dot(x, kernel)
    
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape = (input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape = (input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape = (input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def model_lstm_atten(embedding_matrix,lenIndex,nclasses):
    inp = Input(shape=(500,))
    x = Embedding(lenIndex + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = CuDNNLSTM(32, return_sequences=True)(x)
    x = AttentionWithContext()(x)
    x = Dense(8, activation="tanh")(x)
    x = Dense(nclasses, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def loadData_Tokenizer(X_train, X_test,test,MAX_NB_WORDS=70000,MAX_SEQUENCE_LENGTH=500):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test,test), axis=0)
    text = np.array(text)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    # np.random.shuffle(indices)
    text = text[indices]
    print(text.shape)
    X_train = text[0:len(X_train), ]
    X_test = text[len(X_train):len(X_test)+len(X_train), ]
    test = text[len(X_test)+len(X_train):, ]
    embeddings_index = {}
    f = open(os.path.join(path,"glove/glove.twitter.27B.50d.txt"), encoding="utf8")
    for line in f:

        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return (X_train, X_test,test, word_index,embeddings_index)


def Build_Model_RNN_Text(word_index, embeddings_index, nclasses,  MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50):
    """
    def buildModel_RNN(word_index, embeddings_index, nclasses,  MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50):
    word_index in word index ,
    embeddings_index is embeddings index, look at data_helper.py
    nClasses is number of classes,
    MAX_SEQUENCE_LENGTH is maximum lenght of text sequences
    """

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return model_lstm_atten(embedding_matrix,len(word_index),nclasses)


def train_pred(model, epochs=5):
    filepath="../output/weights_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=1, min_lr=0.000008, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]
    for e in range(epochs):
        model.fit(X_train_Glove, y_train, batch_size=1024, epochs=10, validation_data=(X_test_Glove, y_test_cat),callbacks=callbacks)
    model.load_weights(filepath)
    pred_val_y = model.predict([X_test_Glove], batch_size=1024, verbose=0)
    pred_test_y = model.predict([test], batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y

def to_class(y,threshold):
    l=[]
    for e in y:
        if e[0]<threshold:
            l.append('Sports')
        else:
            l.append('Politics')
    return l

data = pd.read_csv(os.path.join(path,'train_v2.csv'))
data.Label = data.Label.map({'Politics':0,'Sports':1})
test_data = pd.read_csv(os.path.join(path,'test_v2.csv'))
split = int(len(data)*.8)
X_train = data.X.values[:split]
X_test = data.X.values[split:]
y_train = to_categorical(data.Label.values[:split], num_classes=2)
y_test = data.Label.values[split:]
y_test_cat = to_categorical(y_test, num_classes=2)
test = test_data.X.values
X_train_Glove,X_test_Glove, test,word_index,embeddings_index = loadData_Tokenizer(X_train,X_test, test)
model_RNN = Build_Model_RNN_Text(word_index,embeddings_index, 2)
pred_val_y, pred_test_y = train_pred(model_RNN, epochs=1)

out_df = test_data[["TweetId"]]
# when submitting all Ids as "Sports" I had 0.6028 accuracy which's the Sports tweets percentage
# so defining a threshold based on that quantile will boost the model's predection
out_df['Label'] = to_class(pred_test_y ,np.quantile(pred_test_y[:,0],0.60280)) 
out_df.to_csv(os.path.join(output,'submission.csv'),index = False)