from django.urls import path
from .views import UploadImageView

urlpatterns = [
    path('detect/', UploadImageView.as_view()),
]