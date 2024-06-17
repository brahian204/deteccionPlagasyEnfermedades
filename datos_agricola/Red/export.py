from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train10/weights/best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx')