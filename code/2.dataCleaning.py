## Data cleaning
import numpy as np 
import pandas as pd
import string
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import os
kaggle = False
path =''
if kaggle:
	path ='../input'
else:
	path = '../data'

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
def tokenization(text):
    text = re.split('\W+', text)
    return text
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text
def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text
train_df = pd.read_csv(os.path.join(path,'train.csv'))
test_df = pd.read_csv(os.path.join(path,"test.csv"))
test_df['Label'] = np.nan
df = pd.concat([train_df,test_df],axis = 0)
df['TweetText'] = df['TweetText'].apply(lambda x: remove_punct(x))
df['Tweet_tokenized'] = df['TweetText'].apply(lambda x: tokenization(x.lower()))
stopword = nltk.corpus.stopwords.words('english')
df['Tweet_nonstop'] = df['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))
ps = nltk.PorterStemmer()
df['Tweet_stemmed'] = df['Tweet_nonstop'].apply(lambda x: stemming(x))
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
df['Tweet_lemmatized'] = df['Tweet_nonstop'].apply(lambda x: lemmatizer(x))
df['X'] = df.Tweet_lemmatized.apply(lambda x:' '.join(x))
train = df.loc[df.Label.isin(['Politics','Sports'])]
test = df.loc[~df.Label.isin(['Politics','Sports'])]
train.to_csv(os.path.join(path,'train_v2.csv'),index = False)
test.to_csv(os.path.join(path,'test_v2.csv'),index = False)