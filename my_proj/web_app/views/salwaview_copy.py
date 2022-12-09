import numpy as np
import pandas as pd
import os
"""import torch.nn as nn
import torch
import torch.nn.functional as F
import transformers
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset,DataLoader
from torch.autograd import Variable"""
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import tqdm
from sklearn.preprocessing import label_binarize
from matplotlib._path import (affine_transform, count_bboxes_overlapping_bbox, update_path_extents)
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import nltk
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

warnings.filterwarnings('ignore')
nltk.download('stopwords')
stop_words= set(stopwords.words('english'))
nltk.download('punkt')
print(f'all required data is downloaded')

def hate_copy(request, *args, **kwargs):
#dataset=pd.read_csv('C:\\Users\\Sameer\\Hate-Speech-Recognition-main\\resampled_dataset.csv')
    dataset=pd.read_csv('web_app/views/labeled_data.csv')
    #dataset=pd.read_csv(".labeled_data.csv")
    df =dataset.dropna(inplace = True)
    dataset.groupby('class')['id'].nunique().plot(kind='bar',title='Plot of number of tweets belonging to a particular class')
    dataset(df)
def clean_tweet(request, *args, **kwargs):
    tweet = request.POST.get('textA')
    print(tweet)
    tweet = re.sub("#", "",tweet) # Removing '#' from hashtags
    tweet = re.sub("[^a-zA-Z#]", " ",tweet) # Removing punctuation and special characters
    tweet = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',"<URL>", tweet)
    tweet = re.sub('http','',tweet)
    tweet = re.sub(" +", " ", tweet)
    tweet = tweet.lower()
    tweet = word_tokenize(tweet)
    return_tweet=[]
    for word in tweet:
        if word not in stop_words:
            return_tweet.append(word)
    print(return_tweet)
    #return return_tweet
    dataset["tweet"]=dataset["tweet"].apply(clean_tweet)

    for i in range(0,len(dataset["tweet"])):
        dataset["tweet"][i] = " ".join(dataset["tweet"][i])
    X = dataset["tweet"]
    Y = dataset["class"]

    model = Word2Vec(dataset["tweet"].values,  window=5, min_count=1, workers=4)

def get_features(tweet):
    features=[]
    for word in tweet:
        features.append(model.wv[word])
    #return np.mean(features,0)
    dataset["features"]=dataset["tweet"].apply(get_features)
    data=[]
    for i in dataset["features"].values:
        temp=[]
        for j in i:
            temp.append(j)
        data.append(temp)
    data=np.array(data)
    data
    x_train,x_test,y_train,y_test = train_test_split(data,Y,test_size=0.2,random_state=42)
    svm_clf = OneVsRestClassifier(svm.SVC(gamma='scale', probability=True))
    svm_clf.fit(x_train,y_train)
    filename = 'finalized_model.pickle'
    pickle.dump(svm_clf, open(filename, 'wb'))
    y_pred = svm_clf.predict(x_test)
    f = f1_score(y_test, y_pred, average='micro')
    print("F1 Score: ", f)
    p = precision_score(y_test, y_pred, average='micro')
    print("Precision Score: ", p)
    r = recall_score(y_test, y_pred, average='micro')
    print("Recall Score: ", r)
    print("Accuracy: ", svm_clf.score(x_test,y_test))

    #a = input("enter the tweet")
    a = input().split(" ")
    string =' '.join(a)
    model = Word2Vec(string,  window=5, min_count=1, workers=4)
    features=[]
    for word in string:
        features.append(model.wv[word])
    da = np.mean(features,0)

    pre_model = pickle.load(open('C:\\Users\\Sameer\\Hate-Speech-Recognition-main\\finalized_model.pickle', 'rb'))
    print("model loaded")
    result = pre_model.predict(lst1)
    print(result)
    if result == 2:
        print("Neutral tweet")
    elif result == 1:
        print("Hate or offensive tweet")
    elif result == 0:
        print("It is both hate and offensive tweet")













