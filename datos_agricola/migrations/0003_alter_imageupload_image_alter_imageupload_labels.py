# Generated by Django 4.2.10 on 2024-03-06 21:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('datos_agricola', '0002_alter_imageupload_image_alter_imageupload_labels'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageupload',
            name='image',
            field=models.ImageField(upload_to=''),
        ),
        migrations.AlterField(
            model_name='imageupload',
            name='labels',
            field=models.FileField(upload_to=''),
        ),
    ]