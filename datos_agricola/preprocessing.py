from .filtros import filtros, warmImage, retro
import os
import numpy as np
import cv2
import random
import shutil
from PIL import Image


def rename_files(directory, new_name):
    cont = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(('.png', '.JPG', '.jpeg')):
            cont += 1 
            extension = os.path.splitext(filename)[1].lower()
            new_filename = f"{new_name}_{cont}{extension}"
            new_filepath = os.path.join(directory, new_filename)
            os.rename(filepath, new_filepath)
            label_filename = filename.replace('.JPG', '.txt') #.replace('.jpeg', '.txt').replace('.png', '.txt')
            label_filepath = os.path.join(directory, label_filename)
            if os.path.isfile(label_filepath):
                extensionl = os.path.splitext(label_filename)[1]
                new_label_filename = f"{new_name}_{cont}{extensionl}"
                new_label_filepath = os.path.join(directory, new_label_filename)
                os.rename(label_filepath, new_label_filepath)
        print('archivo renombrado') 

def split_data(dpe, ruta_raiz):
    
    dpe = os.path.join(ruta_raiz, dpe)
    # Crear subcarpetas para train y validation
    ruta_train = os.path.join(dpe, "train")
    ruta_validation = os.path.join(dpe, "valid")
    os.makedirs(ruta_train, exist_ok=True)
    os.makedirs(ruta_validation, exist_ok=True)

    # Crear subcarpetas "images" y "labels" dentro de train y validation
    train_images = os.path.join(ruta_train, "images")
    train_labels = os.path.join(ruta_train, "labels")
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)

    validation_images = os.path.join(ruta_validation, "images")
    validation_labels = os.path.join(ruta_validation, "labels")
    os.makedirs(validation_images, exist_ok=True)
    os.makedirs(validation_labels, exist_ok=True)

    # Porcentaje de datos para train (80%) y validation (20%)
    porcentaje_train = 0.8

    # Obtener todas las imágenes y etiquetas
    imagenes = [file for file in os.listdir(ruta_raiz) if file.endswith(('.JPG', '.jpg', '.png'))]
    etiquetas = [file for file in os.listdir(ruta_raiz) if file.endswith(".txt")]

    # Calcular la cantidad de datos para train y validation
    cantidad_train = int(len(imagenes) * porcentaje_train)
    cantidad_validation = len(imagenes) - cantidad_train

    # Seleccionar aleatoriamente los datos para train
    datos_train = random.sample(list(zip(imagenes, etiquetas)), cantidad_train)

    # Mover las imágenes y etiquetas seleccionadas a las carpetas correspondientes
    for imagen, etiqueta in datos_train:
        shutil.move(os.path.join(ruta_raiz, imagen), os.path.join(train_images, imagen))
        shutil.move(os.path.join(ruta_raiz, etiqueta), os.path.join(train_labels, etiqueta))

    # Mover los datos restantes a la carpeta de validation
    for imagen, etiqueta in zip(imagenes, etiquetas):
        if not os.path.exists(os.path.join(train_images, imagen)):
            shutil.move(os.path.join(ruta_raiz, imagen), os.path.join(validation_images, imagen))
            shutil.move(os.path.join(ruta_raiz, etiqueta), os.path.join(validation_labels, etiqueta))
    
    print("Datos divididos correctamente en train y valid")
    
    return train_images, validation_images

    
