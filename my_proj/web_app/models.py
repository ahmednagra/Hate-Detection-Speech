from django.db import models
from django.utils import timezone
from django.urls import reverse
# Create your models here.

"""class hate_spech(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    date_posted = models.DateTimeField(default=timezone.now)
    # author mein user foreign key se add aur delete krny k liya parameter mein on_delete add kiya
    author =  models.ForeignKey(User, on_delete=models.CASCADE)"""