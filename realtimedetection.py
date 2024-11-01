import ultralytics
from ultralytics import YOLO
import cv2
import math

# Open the video file
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
# Model
model = YOLO("C:/Users/saisi/Downloads/best.pt") #add your file path

# Object classes
classNames = [
    "person", "car", "chair", "bottle", "pottedplant", "bird", "dog", "sofa",
    "bicycle", "horse", "boat", "motorbike", "cat", "tvmonitor", "cow", "sheep",
    "aeroplane", "train", "diningtable", "bus"
]

while cap.isOpened():
    # Read a frame from the video
    success, img = cap.read()
    if not success:
        break

    # Perform object detection
    results = model(img, stream=True)

    # Process detection results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Draw bounding box on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = (math.ceil((box.conf[0] * 100)) / 100)*100

            # Class name
            cls = int(box.cls[0])

            # Object details
            org = (x1, y1 - 10)  # Adjusting the position for displaying text
            font = cv2.FONT_HERSHEY_DUPLEX
            fontScale = 0.5  # Adjust font size if needed
            color = (100, 0, 128)  # White color for text
            thickness = 2  # Thickness of text

            # Display class name and confidence on the bounding box
            text = f"{classNames[cls]}: {confidence:.2f}"
            cv2.putText(img, text, org, font, fontScale, color, thickness)

    # Display the frame
    cv2.imshow('Video', img)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
