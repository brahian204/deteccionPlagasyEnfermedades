import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('C:/Users/BRAHIAN SANCHEZ/Documents/data_Django/image_acquisition/models/detect_plg.pt')  # load a custom model

# Open the video file
video_path = "C:/Users/BRAHIAN SANCHEZ/Documents/data_Django/image_acquisition/datos_agricola/Red/test/fly.mp4"
cap = cv2.VideoCapture(video_path)

# Obtener la frecuencia de cuadros del video
fps = cap.get(cv2.CAP_PROP_FPS)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Calcular el tiempo de espera para mantener la velocidad normal
        wait_time = int(10000 / fps)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()