
from django.db import models

class DataUpload(models.Model):
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
    dpe = models.CharField(max_length=255, default='dpe')
    images = models.ImageField()
    labels = models.FileField()
    fecha_carga = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=50,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    
class DataTrain(models.Model):
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
    dpe = models.ForeignKey(DataUpload, on_delete=models.CASCADE, related_name='train')
    name_model = models.CharField(max_length=50)
    num_epoch = models.CharField(max_length=10)
    fecha_train = models.DateTimeField(auto_now_add=True)
    num_data = models.CharField(max_length=10)
    status = models.CharField(
        max_length=50,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    