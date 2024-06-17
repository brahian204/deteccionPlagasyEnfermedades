from rest_framework import serializers
from .models import DataUpload

class DataUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataUpload
        # exclude = ('dpe', )
        fields = ('images', 'labels')
        
    