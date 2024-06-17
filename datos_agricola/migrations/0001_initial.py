# Generated by Django 4.2.10 on 2024-03-01 20:12

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ImageUpload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('dpe', models.CharField(default='plaga', max_length=255)),
                ('image', models.ImageField(upload_to='data/images')),
                ('labels', models.FileField(upload_to='data/labels')),
                ('fecha_carga', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]