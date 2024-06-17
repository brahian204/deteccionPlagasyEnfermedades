from django.shortcuts import render
from django.http.response import JsonResponse
from rest_framework import status
from django.http import HttpResponse
from rest_framework.decorators import api_view
from django.conf import settings
from rest_framework.parsers import JSONParser
import os
from ultralytics import YOLO
import cv2
from database import *
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from .preprocessing import split_data
from .models import DataUpload, DataTrain
from .serializer import DataUploadSerializer
from .validacion import validate_files, validacion_fotos
from .Red.train import train
from .augmentation_album.main import *

@csrf_exempt
@api_view(['POST'])
def upload_data(request):
    if request.method == 'POST':
        dpes = request.POST.get('dpe')
        images = request.FILES.getlist('images')
        labels = request.FILES.getlist('labels')
        
        if len(images) != len(labels):
            print(f'La cantidad de archivos no coinciden, enviaste: {len(images)} imagenes y {len(labels)} etiquetas')
            return JsonResponse({'message': 'Cantidad de archivos no coinciden','code':-1}, status = status.HTTP_400_BAD_REQUEST)
        
        for img, lbl in zip(images, labels):             
            data = {'dpe': dpes, 'images': img, 'labels': lbl}
            serializer = DataUploadSerializer(data=data)
            
            if serializer.is_valid():
                info = serializer.save()
                info.status = DataUpload.STATUS_COMPLETED
                info.save()
                print(f'Datos guardados correctamente: {img}, {lbl}', '\n')
            else:
                print(f'Error al guardar datos: {serializer.errors}', '\n')

        path_data = settings.MEDIA_ROOT
        
        if validacion_fotos(path_data):
            # Transformaciones
            rotaciones()
            filtros(path_data)

            # Division de datos: 80% train 20% valid
            paths = split_data(dpes, path_data)
            path_train, path_valid = paths
            
            canttrain = len(os.listdir(path_train))
            cantval = len(os.listdir(path_valid))
            
            print('AUMENTO DE DATOS REALIZADO CON EXITO', '\n',
                  'TOTAL DATOS PARA TRAIN: ', canttrain, '\n',
                  'TOTAL DATOS PARA VALID: ', cantval)
        
        return JsonResponse({'message': 'proceso exitoso','code':1}, status = status.HTTP_200_OK)
    else:
        return JsonResponse({'message': 'debes enviar algun dato','code':-1}, status = status.HTTP_400_BAD_REQUEST)

@csrf_exempt
@api_view(['POST'])
def train_data(request):
    if request.method == 'POST':
        
        try:
            data = train()
            
            print(data)
        except Exception as error:
            print('error: ', error)
