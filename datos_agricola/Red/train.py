from ultralytics import YOLO
import torch
import multiprocessing

def train():

    multiprocessing.freeze_support()  # Esta línea es necesaria en Windows

    # Configura el dispositivo como GPU si está disponible, de lo contrario, utiliza CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Versión de CUDA disponible:", torch.version.cuda)

    # Load a model
    model = YOLO('datos_agricola/Red/yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='datos_agricola/Red/custom_data.yaml', epochs=5, imgsz=512, batch=8 )
    
    return results
