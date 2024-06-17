from django.http.response import JsonResponse
from rest_framework import status
import os
import cv2
from ultralytics import YOLO
import torch
import multiprocessing
import os
import shutil
import numpy as np
from datos_agricola.Red.train import train
from .preprocessing import rename_files


def validacion_fotos(path_img):
    image_files = [f for f in os.listdir(path_img) if f.endswith(('.jpg','.JPG', '.jpeg', '.png'))]
    
    # rename_files(pathtrain, dpe)

    if len(image_files) >= 1:
        # print(f'Cantidad de imagenes aceptada para el entrenamiento: {canttrain} y validacion: {cantval}',)
        print(f'Cantidad de imagenes aceptada para el aumento de datos', '\n') 
        return True
        
    else: 
        print(f'datos insuficientes para el aumento de datos')  
        return False   
        

def validate_files(img, label):
    try:
        for i, l in zip(img, label):
            if i.content_type.startswith('image/'):
                if i.filename.lower().endswith('.jpg') or i.filename.lower().endswith('.jpeg'):
                    continue
                else:
                    print('formato incorrecto, cargue una imagen')
                    return JsonResponse({'message': 'Formato incorrecto, cargue una imagen','code':-1}, status = status.HTTP_400_BAD_REQUEST)
            elif l.content_type.startswith('text/'):
                if l.filename.lower().endswith('.txt'):
                    continue
                else:
                    print('formato incorrecto, cargue un txt')
                    return JsonResponse({'message': 'Formato incorrecto, cargue un archivo txt','code':-1}, status = status.HTTP_400_BAD_REQUEST)
            else:
                return JsonResponse({'message': 'Formato incorrecto','code':-1}, status = status.HTTP_400_BAD_REQUEST)
    
    except Exception as error:
            print('ocurrio un error con los archivos cargados: ', error)
    # Validate file existence
    # if not os.path.isfile(img) or not os.path.isfile(label):
    #     return False
    
    
        