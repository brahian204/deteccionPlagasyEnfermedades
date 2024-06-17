from django.urls import path
from .views import upload_data, train_data

urlpatterns = [
    path('upload/', upload_data),
    path('train/', train_data),
]