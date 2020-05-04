from django.urls import path, re_path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'filename_database'

urlpatterns = [
    path('', views.main_page, name='main_page'),
]