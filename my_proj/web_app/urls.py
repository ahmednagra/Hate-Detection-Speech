
from django.urls import path
from web_app.views.views import *
from web_app.views.det2 import *
from web_app.views.client import *

urlpatterns = [
   
    path('', home, name = 'home'),
    path('overview/', overview, name = 'overview'),
    path('overview/screen_tweet', hate_speech, name ='det2'),
    path('traindata/',train_data, name = 'traindata'),
    path('testdata/',test_data, name = 'testdata'),
    path('stat/', statistics, name = 'stat'),
    path('about/', about, name = 'about'),
]
