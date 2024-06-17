# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO('models/train_all/best.pt')

# # Export the model to ONNX format
# model.export(format='onnx')  # creates 'yolov8n.onnx'

# # Load the exported ONNX model
# onnx_model = YOLO('best (13).pb')

# # Run inference
# results = onnx_model('datos_agricola/Red/test/fly3.jpg', show=True,save=False, imgsz=512)

import torch
from onnx2torch.converter import convert
import onnx

# Path to ONNX model
onnx_model_path = 'insect.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# Imprime un mensaje de éxito
print(f"El modelo ONNX en {onnx_model_path} se ha convertido exitosamente a PyTorch.")
