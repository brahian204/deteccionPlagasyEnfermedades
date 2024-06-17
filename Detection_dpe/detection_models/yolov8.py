from ultralytics import YOLO
import cv2

class YOLOv8Detector:
    def __init__(self, model_path):
        """
        Initialize the YOLOv8 model.
        :param model_path: The path to the YOLOv8 model file (e.g., 'yolov8n.pt').
        """
        # Load the pretrained YOLOv8 model
        self.model = YOLO(model_path)

    def apply_nms(self, boxes, scores, threshold=0.6):
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=threshold, nms_threshold=0.5)
        selected_boxes = [boxes[i[0]] for i in indices]
        return selected_boxes
    
    
    def run_detection(self, image_path, confidence_threshold=0.8):
        """
        Run object detection on the provided image.
        :param image_path: The path to the image file.
        :return: A list of detected objects with their labels, confidence scores, and bounding boxes.
        """
        # Perform detection
        results = self.model(image_path, conf=confidence_threshold, stream=False)[0]

        # Process results
        detected_objects = []
        for det in results.boxes:  # Loop through each detection
            label_index = int(det.cls)
            label = self.model.names[label_index]
            confidence = float(det.conf)
            bbox = det.xyxy.cpu().numpy()[0]
            x_min, y_min, x_max, y_max = bbox[:4]

            detected_objects.append({
                'label': label,
                'confidence': confidence,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
            })
            
         # Apply NMS
        # boxes = [[obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']] for obj in detected_objects]
        # scores = [obj['confidence'] for obj in detected_objects]
        # selected_boxes = self.apply_nms(boxes, scores)

        return detected_objects #, selected_boxes

        # return detected_objects