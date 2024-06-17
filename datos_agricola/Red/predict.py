from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('C:/Users/BRAHIAN SANCHEZ/Documents/data_Django/image_acquisition/datos_agricola/Red/models/flyaug.pt')  # load a custom model

# Read an image using OpenCV
source = cv2.imread('C:/Users/BRAHIAN SANCHEZ/Documents/data_Django/image_acquisition/datos_agricola/Red/test/image1.png')

alto, ancho, _ = source.shape

# Run inference on the source, Predict with the model
results = model(source, show=True, conf=0.8,save=False, )  # list of Results objects

cv2.waitKey(0)#Esto para que no estas ploteando ninguna imagen 
cv2.destroyAllWindows() #Esto para que no estas ploteando ninguna imagen x2


for r in results:
    for box in r.boxes:
        x, y, w, h = box.xyxy[0]  # coordenadas x, y, ancho y alto
        
        x_N = x / ancho
        y_N = y / alto

        # Para convertir el ancho y el alto a valores normalizados:
        w_ = w / ancho
        h_ = h / alto
        
        with open("valores.txt", "a") as archivo:
        # Escribir los valores en el archivo
            archivo.write(f"0 {x_N} {y_N} {w_} {h_} \n")
            print("Valores escritos en el archivo 'valores.txt'")

        # archivo.write(f"Ancho: {w_}\n")
        # archivo.write(f"Coordenada X: {x_N}\n")
        # archivo.write(f"Coordenada Y: {y_N}\n")
