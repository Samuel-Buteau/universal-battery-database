from django.contrib import admin
from django.urls import path
from . import views

from django.conf.urls import url


app_name = 'WetCellDatabase'

urlpatterns = [
    path('', views.define_page, name='define_page'),

]

