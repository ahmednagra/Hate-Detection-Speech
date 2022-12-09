from django.shortcuts import render, get_object_or_404
from datetime import datetime
from web_app.views.det2 import *
# Create your views here.

def home(request):
    return render(request, 'web_app/index.html')

def overview(request):
    return render(request, 'web_app/overview.html')



def train_data(request):
    return render(request, 'web_app/traindata.html')

    

def test_data(request):
    return render(request, 'web_app/test.html')


def statistics(request):
    return render(request, 'web_app/statistics.html')   


def about(request):
    return render(request, 'web_app/index.html')


"""class overview(TemplateView):
    template_name = "web_app/overview.html"
    
    def get(self, request, **kwargs):
        form = ParserForm()
        return render(request, self.template_name, {"form": form})
    
    def post(self, request, **kwargs):
        form = ParserForm(request.POST)

        if form.is_valid():
            inputtext = form['InputText'].value()
            template = form['InputTemplate'].value()
            # Process the data and get the result
            print(result)
            return render(request, self.template_name, {'result': result})
    
    #return render(request, 'web_app/overview.html')
"""
