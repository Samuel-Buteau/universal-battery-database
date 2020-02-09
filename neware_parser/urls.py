"""FTIR_to_electrolyte_composition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path, include
from .views import *

app_name = 'neware_parser'
urlpatterns = [
    re_path(r'^view_barcode/(?P<barcode>\d+)/(?P<cursor>[A-Z_]*)/$', view_barcode, name='view_barcode'),
    path('overview/', main_page, name='main_page'),
    path('', index, name='index'),
    path('admin/', admin.site.urls),
    path('FileNameHelper/', include('FileNameHelper.urls')),
    path('WetCellDatabase/', include('WetCellDatabase.urls')),
]
