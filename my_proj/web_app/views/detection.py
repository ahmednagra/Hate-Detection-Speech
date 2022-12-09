from django.shortcuts import render, get_object_or_404

#removig stop words just
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def hate_ss(request, *args, **kwargs):
    if request.method == 'POST':
        sentence = request.POST.get('textA')
        print(sentence)       
        # Lower case all the words of the tweet before any preprocessing
        df=(sentence.lower())
        print(df)
        
        new_string = []
        word = [word for word in df if word not in stopwords.words('english')]
        return "" . join(new_string)
        
    return render(request, "web_app/overview.html", {'text':new_string }, {'old_text': sentence})    