
from django.urls import path
from web_app.views.views import *
from web_app.views.detection import *
from web_app.views.det2 import *
from web_app.views.salwaview import *
from web_app.views.salwaview_copy import *

urlpatterns = [
   
    path('', home, name = 'home'),
    path('overview/', overview, name = 'overview'),
    path('overview/clean_tweet', clean_tweet, name ='det2'),
    path('traindata/',train_data, name = 'traindata'),
    path('testdata/',test_data, name = 'testdata'),
    path('stat/', statistics, name = 'stat'),
    path('about/', about, name = 'about'),
]
