from django.contrib import admin
from django.urls import path
from . import views

from django.conf.urls import url


app_name = 'WetCellDatabase'

urlpatterns = [
    url(r'^define/(?P<mode>\w+)/$', views.define_page, name='define_page'),
    path('search/', views.search_page, name='search_page'),
    url(r'^define_wet_cell_bulk/(?P<predefined>\w+)/$', views.define_wet_cell_bulk, name='define_wet_cell_bulk'),
    path('delete/', views.delete_page, name='delete_page'),

]

