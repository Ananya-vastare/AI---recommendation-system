from django.urls import path
from . import views

urlpatterns = [
    path('search/', views.server, name='server'),
]