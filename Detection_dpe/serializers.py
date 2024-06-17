from rest_framework import serializers
from .models import ImagesUpload

class ImageUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImagesUpload
        fields = ['image_file', 'confidence_threshold']