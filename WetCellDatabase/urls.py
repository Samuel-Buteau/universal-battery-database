from django.contrib import admin
from django.urls import path
from . import views

from django.conf.urls import url


app_name = 'WetCellDatabase'

urlpatterns = [
    path('define/', views.define_page, name='define_page'),
    path('search/', views.search_page, name='search_page'),
]

