from datos_agricola.filtros import filtros
import os
import numpy as np
import cv2
import albumentations as A
from scipy.interpolate import UnivariateSpline
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image, ImageFilter,ImageEnhance
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import shutil
import random

def rotate_images(directorio, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Crear un ImageDataGenerator para las rotaciones
    datagen = ImageDataGenerator(
        rotation_range=90,
        fill_mode='nearest'
    )

    # Iterar sobre todas las imágenes en el directorio de entrada
    for filename in os.listdir(directorio):
        if filename.endswith(('.JPG', '.jpeg', '.png')):
            img_path = os.path.join(directorio, filename)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Generar y guardar las imágenes rotadas
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=filename.split('.')[0], save_format='jpg'):
                i += 1
                if i >= 4:  # Rotar 90 grados a la izquierda, 90 grados a la derecha, 45 grados y 180 grados
                    print('imagen rotada con exito')
                    break

def augment_data(input_dir, output_dir):
    
    # Crea el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista todos los archivos de imagen en el directorio de entrada
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.JPG', '.jpeg', '.png'))]

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path)

        # Rota 90° a la izquierda
        img_left = img.rotate(90)
        img_left_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_90.jpg")
        img_left.save(img_left_path)

        # Rota 90° a la derecha
        img_right = img.rotate(-90)
        img_right_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_-90.jpg")
        img_right.save(img_right_path)

        # Rota 180°
        img_180 = img.rotate(180)
        img_180_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_180.jpg")
        img_180.save(img_180_path)
        
        print('imagenes rotadas con exito')

def filtros_image(ruta,img, dpe):

    lista = filtros(img)
    for image in range(len(lista)):
        path_image = str (ruta+'/{}_{}'.format(dpe,len(os.listdir(ruta)))+'.jpg')    
        print('imagen filtrada: ', path_image)
        cv2.imwrite(path_image,lista[image])
        
def warmImage(img):
    img = img.astype(np.uint8)
    def spreadLookupTable(x, y):
        spline = UnivariateSpline(x, y)
        return spline(range(256))
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    warmImage = cv2.merge((red_channel, green_channel, blue_channel))   
    red_channel, green_channel, blue_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    coldImage = cv2.merge((red_channel, green_channel, blue_channel))
    return warmImage,coldImage

def exponential_function(channel, exp):
    table = np.array([min((i**exp), 255) for i in np.arange(0, 256)]).astype("uint8") # creating table for exponent
    channel = cv2.LUT(channel, table)
    return channel

def tone(image, number):
    for i in range(3):
        if i == number:
            image[:, :, i] = exponential_function(image[:, :, i], 0.85) # applying exponential function on slice
        else:
            image[:, :, i] = 0 # setting values of all other slices to 0
    return image

def brillo(image):
     cols, rows, channel = image.shape
     brightness = np.sum(image) / (255 * cols * rows * channel)
     minimum_brightness = 0.6
     alpha = brightness / minimum_brightness
     bright_img = cv2.convertScaleAbs(image, alpha = alpha, beta = 1 * (1 - alpha))
     bright_img1 = cv2.convertScaleAbs(image, alpha = 0.8, beta = 15)    
     return bright_img1,bright_img 
 
def histograma(image):
     image = image.astype(np.uint8)    
     img_to_yuv = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
     img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])  
     img_to_yuv = cv2.cvtColor(img_to_yuv,cv2.COLOR_YUV2RGB)   
     return img_to_yuv

def serpia(image):
    image = np.array(image, dtype=np.float64) # converting to float to prevent loss
    image = cv2.transform(image, np.matrix([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                   [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    image[np.where(image > 255)] = 255 # normalizing values greater than 255 to 255
    image = np.array(image, dtype=np.uint8)    
    return image 

def laplace(foto):
    w = 1/3
    foto = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
    foto = Image.fromarray(foto.astype(np.uint8))
    coeficientes = [1, 1, 1, 1, -8, 1, 1, 1, 1]
    datos_laplace = foto.filter(ImageFilter.Kernel((3,3), coeficientes, 1)).getdata()
    datos_imagen = foto.getdata()
    datos_nitidez = [datos_imagen[x] - (w * datos_laplace[x]) for x in range(len(datos_laplace))]   
    imagen_nitidez = Image.new('L', foto.size)
    imagen_nitidez.putdata(datos_nitidez)  
    imagen_nitidez = np.array(imagen_nitidez, dtype=np.uint8)
    imagen_nitidez = np.reshape(imagen_nitidez,(512,512,1))
    return imagen_nitidez 

def apertura(img):
    img = img.astype(np.uint8)
    img_to_yuv = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    contraste = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2RGB)
    return contraste

def salt_pepper(prob,gray,image):
    s_vs_p = 0.05
    output = np.copy(image)
    num_salt = np.ceil(prob * gray.size * (s_vs_p))
    coords = [np.random.randint(0, i-1, int(num_salt))for i in gray.shape]
    output[tuple(coords)] = 0
    num_salt = np.ceil(prob * gray.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1, int(num_salt))for i in gray.shape]
    output[tuple(coords)] = 1
    return output 

def retro(img):
     rows, cols = img.shape[0:2]
     kernel_x = cv2.getGaussianKernel(cols,200)
     kernel_y = cv2.getGaussianKernel(rows,200)
     kernel = kernel_y * kernel_x.T
     filter = 255 * kernel / np.linalg.norm(kernel)  
     vintage_im = np.copy(img) 
     for i in range(3):
         vintage_im[:,:,i] = vintage_im[:,:,i] * filter 
     return vintage_im 

def color (image,number):
      image = image.astype(np.uint8)
      image = tone(image,number) 
      return image 
  
def comic(img):
     img_invert = cv2.bitwise_not(img)
     img_smoothing = cv2.GaussianBlur(img_invert, (21,21), sigmaX=0, sigmaY=0)
     def dodge(x,y):
         return cv2.divide(x, 255-y, scale=256)
     final_img=dodge(img, img_smoothing)   
     return final_img 
 
def cartoon(image):    
     image = image.astype(np.uint8)
     img_invert = cv2.bitwise_not(image)
     edges1 = cv2.bitwise_not(cv2.Canny(image, 110, 220)) 
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     gray = cv2.medianBlur(gray, 5) 
     edges2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3,3) 
     dst = cv2.edgePreservingFilter(image, flags=2, sigma_s=64, sigma_r=0.25) 
     cartoon1 = cv2.bitwise_and(dst, dst, mask=edges1)
     cartoon2 = cv2.bitwise_and(dst, dst, mask=edges2)
     return cartoon2,cartoon1,img_invert

def imadjust(img, In=(0,1.0), Out=(0,1.0), gamma=1.0):
    "J = low_out +(high_out - low_out).* ((I - low_in)/(high_in - low_in)).^ gamma"
    low_in,high_in = In
    low_out, high_out = Out
 
    low_in *= 255.0
    high_in *= 255.0
 
    low_out *= 255.0
    high_out *= 255.0    
    
    k = (high_out - low_out) / (high_in - low_in)
         # Gamma transformation table
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    h,w = img.shape[:2]
    imgOut = np.zeros((h,w), np.uint8)
    
    for r in range(h):
        for c in range(w):
            if img[r,c] <= low_in:
                imgOut[r,c] = low_out                
            elif img[r,c] > high_in:
                imgOut[r,c] = high_out
            else:
                res = int(k*(img[r,c]-low_in) + low_out)
                imgOut[r,c] = table[res]#Check table
               
    return imgOut    

def backgroud(img):
    kernel = np.ones((5,5),np.uint8)
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    I23 = img - background
    I23 = imadjust(I23,(0,1), (0,1), 0.5)  
    rangefiltG = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    rangefiltG = imadjust(rangefiltG,(0,1), (0,1), 0.5) 
    return I23,rangefiltG 

def ImgEnhace(img):
    img = img.astype(np.uint8)
    color = ImageEnhance.Color(Image.fromarray(img))
    color = color.enhance(0.8)
    color =ImageEnhance.Sharpness(color)
    color = color.enhance(0.8)
    color = ImageEnhance.Contrast(color)
    color = color.enhance(1.2)
    return np.array(color).astype('uint8')

def warmImage(img):
    img = img.astype(np.uint8)
    def spreadLookupTable(x, y):
        spline = UnivariateSpline(x, y)
        return spline(range(256))
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 90, 180, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    warmImage = cv2.merge((red_channel, green_channel, blue_channel))   
    red_channel, green_channel, blue_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    coldImage = cv2.merge((red_channel, green_channel, blue_channel))
    return warmImage,coldImage 

def apply_filter(image_path, output_path):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Iterar sobre cada archivo en la ruta
    for filename in os.listdir(image_path):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            # Cargar la imagen
            image = cv2.imread(os.path.join(image_path, filename))
            
            # Aplicar todos los filtros disponibles
            filters = ['sharpness', 'warm', 'cool', 'grayscale', 'retro']
            for filter_type in filters:
                if filter_type == 'sharpness':
                        # Filtro de nitidez
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    filtered_image = cv2.filter2D(image, -1, kernel)
                elif filter_type == 'warm':
                    # Filtro de calido
                    war, filtered_image = warmImage(image)
                elif filter_type == 'cool':
                    # Filtro de frio
                    filtered_image, war = warmImage(image)
                elif filter_type == 'grayscale':
                    # Filtro de escala de grises
                    filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif filter_type == 'retro':
                    # Filtro de anochecer
                    filtered_image = retro(image)

                # Guardar la imagen filtrada en la carpeta de salida
                output_filename = os.path.join(output_path, f'{filter_type}_{filename}')
                print(f'filtro {filter_type} aplicado a la imagen {filename} con exito', '\n')
                cv2.imwrite(output_filename, filtered_image)

def rename_files(directory, new_name):
    cont = 0
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        filepath = os.path.join(directory, filename)
        # Check if the file is an image file
        if filename.endswith(('.png', '.JPG', '.jpeg')):
            cont += 1 
            # Get the extension of the image file
            extension = os.path.splitext(filename)[1].lower()
            # Construct the new filename
            new_filename = f"{new_name}_{cont}{extension}"
            # Construct the new file path
            new_filepath = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(filepath, new_filepath)
            # Check if a corresponding label file exists
            label_filename = filename.replace('.JPG', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            label_filepath = os.path.join(directory, label_filename)
            if os.path.isfile(label_filepath):
                extensionl = os.path.splitext(label_filename)[1]
                # Construct the new label filename
                new_label_filename = f"{new_name}_{cont}{extensionl}"
                # Construct the new label file path
                new_label_filepath = os.path.join(directory, new_label_filename)
                # Rename the label file
                os.rename(label_filepath, new_label_filepath)
        print('archivo renombrado')

# Directorio de imágenes y etiquetas
image_dir = "augdata/train/images"
label_dir = "augdata/train/labels"

# Lista de nombres de archivos (sin extensión)
# file_names = ["imagen1", "imagen2", "imagen3"]  # Agrega tus nombres de archivos aquí
def augdata(image_dir, label_dir):
    cont = 0

    # Transformaciones de Albumentations
    transform = A.Compose([
        A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='yolo'))

    # Procesamiento de cada imagen
    for file_namei, file_namel in zip(os.listdir(image_dir), os.listdir(label_dir)):
        print(file_namei, file_namel, '\n' )
        cont += 1
        # Carga la imagen
        image_path = os.path.join(image_dir, file_namei)
        image = cv2.imread(image_path)

        # Carga las etiquetas (en formato YOLO)
        label_path = os.path.join(label_dir, file_namel)
        with open(label_path, "r") as f:
            lines = f.readlines()
            # bboxes = [list(map(float, line.strip().split())) for line in lines]
            # print(bboxes)
            bboxes = []
            for line in lines:
                parts = line.strip().split()
                class_label = int(parts[0])  # Convierte el primer valor a entero
                x_center, y_center, width, height = map(float, parts[1:])
                bboxes.append([class_label, x_center, y_center, width, height])
            print(bboxes)

        # Aplica las transformaciones
        transformed = transform(image=image, bboxes=bboxes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        # Guarda la imagen transformada
        transformed_image_path = os.path.join(image_dir, f"{cont}_augmented.jpg")
        cv2.imwrite(transformed_image_path, transformed_image)

        # Guarda las etiquetas transformadas (en formato YOLO)
        transformed_label_path = os.path.join(label_dir, f"{cont}_augmented.txt")
        with open(transformed_label_path, "w") as f:
            for bbox in transformed_bboxes:
                f.write(f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")

    print("Aumento de datos completado. Imágenes y etiquetas guardadas.")


    # Ruta de la carpeta raíz donde están las imágenes y etiquetas
    ruta_raiz = ruta
    
    ruta_img, ruta_lab = f'{ruta_raiz}/images', f'{ruta_raiz}/labels' 

    # Crear subcarpetas para train y validation
    ruta_train = os.path.join(ruta_raiz, "train")
    ruta_validation = os.path.join(ruta_raiz, "valid")
    os.makedirs(ruta_train, exist_ok=True)
    os.makedirs(ruta_validation, exist_ok=True)

    # Crear subcarpetas "images" y "labels" dentro de train y validation
    train_images = os.path.join(ruta_train, "images")
    train_labels = os.path.join(ruta_train, "labels")
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)

    validation_images = os.path.join(ruta_validation, "images")
    # validation_labels = os.path.join(ruta_validation, "labels")
    os.makedirs(validation_images, exist_ok=True)
    # os.makedirs(validation_labels, exist_ok=True)

    # Porcentaje de datos para train (80%) y validation (20%)
    porcentaje_train = 0.8

    # Obtener todas las imágenes y etiquetas
    imagenes = [file for file in os.listdir(ruta_img) if file.endswith(".JPG")]
    # etiquetas = [file for file in os.listdir(ruta_lab) if file.endswith(".txt")]

    # Calcular la cantidad de datos para train y validation
    cantidad_train = int(len(imagenes) * porcentaje_train)
    cantidad_validation = len(imagenes) - cantidad_train

    # Seleccionar aleatoriamente los datos para train
    datos_train = random.sample(list(zip(imagenes)), cantidad_train)

    # Mover las imágenes y etiquetas seleccionadas a las carpetas correspondientes
    for imagen in datos_train:
        shutil.move(os.path.join(ruta_raiz, imagen), os.path.join(train_images, imagen))
        # shutil.move(os.path.join(ruta_raiz, etiqueta), os.path.join(train_labels, etiqueta))

    # Mover los datos restantes a la carpeta de validation
    for imagen in imagenes:
        if not os.path.exists(os.path.join(train_images, imagen)):
            shutil.move(os.path.join(ruta_raiz, imagen), os.path.join(validation_images, imagen))
            # shutil.move(os.path.join(ruta_raiz, etiqueta), os.path.join(validation_labels, etiqueta))
            
    print("Datos divididos correctamente en train y valid")
    
    return train_images, validation_images

import os
from matplotlib import pyplot as plt
import cv2

def load_label( DIR):
        labels = []
        with open(DIR) as f:
            for line in f:
                data_inline = line.split(" ")
                label = {
                    "class": int(data_inline[0]),
                    "x_center": float(data_inline[1]),
                    "y_center": float(data_inline[2]),
                    "width": float(data_inline[3]),
                    "height": float(data_inline[4][:-1])
                }
                labels.append(label)
        return labels

def visualize_images_with_bounding_boxes(image_dir, label_dir):
    """
    Display images with bounding boxes.

    Parameters:
    - image_dir (str): Path to the directory containing images.
    - label_dir (str): Path to the directory containing .txt files with bounding box coordinates.

    Returns:
    - None
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path)

        # Load bounding boxes from the corresponding .txt file
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        bounding_boxes = load_label(label_path)
        
        dh, dw, _ = img.shape

        # Draw bounding boxes on the image
        for bb in bounding_boxes:
            # Extract coordinates and draw the rectangle
            x=bb["x_center"]
            y=bb["y_center"]
            w=bb["width"]
            h=bb["height"]
            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)
            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1
            cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 4)

        # Show the image
        plt.figure()
        plt.imshow(img)
        plt.title(image_file)
        plt.axis("off")  # Hide axes

    plt.show()

def split_data(ruta_raiz):
    # Ruta de la carpeta raíz donde están las imágenes y etiquetas
    # ruta_img, ruta_lab = f'{ruta_raiz}/images', f'{ruta_raiz}/labels' 

    # Crear subcarpetas para train y validation
    ruta_train = os.path.join(ruta_raiz, "train")
    ruta_validation = os.path.join(ruta_raiz, "valid")
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
    
    print('images: ', imagenes, '\n')
    print('labels: ', etiquetas, '\n')

    # Calcular la cantidad de datos para train y validation
    cantidad_train = int(len(imagenes) * porcentaje_train)
    cantidad_validation = len(imagenes) - cantidad_train
    
    print('cantidad train: ', cantidad_train)

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

if __name__ == '__main__':
    
    path_data = 'data'
    
    split_data(path_data)
    
    
