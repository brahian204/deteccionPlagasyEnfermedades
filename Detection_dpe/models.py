# Create your models here.
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class ImagesUpload(models.Model):
    STATUS_PENDING = 'pending'
    STATUS_TRAINING = 'training'
    STATUS_UNTRAINED = 'Untrained'
    STATUS_PROCESSING = 'processing'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'

    STATUS_CHOICES = [
            (STATUS_PENDING, 'Pending'),
            (STATUS_TRAINING, 'Training'),
            (STATUS_UNTRAINED, 'Untrained'),
            (STATUS_PROCESSING, 'Processing'),
            (STATUS_COMPLETED, 'Completed'),
            (STATUS_FAILED, 'Failed'),
        ]
    
    image_file = models.ImageField(upload_to='detection/')
    upload_timestamp = models.DateTimeField(auto_now_add=True)
    confidence_threshold = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    status = models.CharField(
        max_length=50,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )


class Detections(models.Model):
    object_detection = models.ForeignKey(ImagesUpload, on_delete=models.CASCADE, related_name='detections')
    label = models.CharField(max_length=100)
    confidence = models.FloatField()
    x_min = models.FloatField()
    x_max = models.FloatField()
    y_min = models.FloatField()
    y_max = models.FloatField()
    detection_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.label} ({self.confidence})"