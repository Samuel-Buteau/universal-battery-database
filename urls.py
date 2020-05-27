"""

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path("", views.home, name="home")
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path("", Home.as_view(), name="home")
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path("blog/", include("blog.urls"))
"""
from django.contrib import admin
from django.urls import path, re_path, include
from cycling.views import index

urlpatterns = [
    
    path("", index, name = "index"),
    path("admin/", admin.site.urls),
    path("filename_database/", include("filename_database.urls")),
    path("cell_database/", include("cell_database.urls")),
    path("cycling/", include("cycling.urls")),
]
