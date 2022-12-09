import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
#punctuation Remove
import re
# Text Pre-processing libraries
import nltk
import string
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from better_profanity import profanity # for offensive words
from django.shortcuts import render, get_object_or_404
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')
stopword=set(stopwords.words('english'))
#print(stopword)
stemmer = nltk. SnowballStemmer("english")
def hate_speech(request, *args, **kwargs):
   
    text = request.POST.get('textA')
    print(text)
    # remove Punctuation 
    text = re.sub(r'[^\w\s]','',text).lower()
    print(text)
    offensive_remove= profanity.censor(text)
    print(offensive_remove)

    imp_words = []
    # Storing the important words
    for word in str(offensive_remove):
        if word not in stopword:
            # Let's Lemmatize the word as well
            # before appending to the imp_words list.
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize(word)
            imp_words.append(word)
    print(imp_words)
    context = {
        'imp_words':imp_words,
        'text':text
        }
    return render(request, 'web_app/overview.html', context )    