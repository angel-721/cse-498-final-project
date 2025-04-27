import torch
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

classes = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd', 'Qh', 'Qs']

model = YOLO("./models/yolo11n-cards-60-epochs.pt")

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


def draw_bounding_box(results, frame):
    # Get frame dimensions directly from the frame
    height, width = frame.shape[:2]

    # Display results
    for result in results:
        # Get the bounding boxes and labels
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        # Draw bounding boxes on the image
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)

            # Ensure coordinates are within frame boundaries
            x1 = max(0, min(width-1, x1))
            y1 = max(0, min(height-1, y1))
            x2 = max(0, min(width-1, x2))
            y2 = max(0, min(height-1, y2))

            # Use different colors for different classes (if multiple classes)
            color = (0, 255, 0)  # Default green
            if hasattr(result, 'names') and result.names:
                # Create a color based on class_id if class names are available
                colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255)]
                color = colors[class_id % len(colors)]

            # Draw rectangle with thickness based on confidence
            thickness = max(1, int(score * 3))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw filled rectangle for text background
            label_text = f'{classes[class_id]}: {score:.2f}'
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)

            # Add text with better contrast
            cv2.putText(frame, label_text, (x1, y1-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def webcam_main():
    while True:
        ret, frame = cam.read()
        # Resize the frame to the input size of the model
        resized_frame = cv2.resize(frame, (640, 640))

        # Perform inference
        results = model.predict(source=resized_frame, conf=0.5, iou=0.5)
        frame = draw_bounding_box(results, frame)

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def picture_main(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Resize the image to the input size of the model
    resized_image = cv2.resize(image, (640, 640))

    # Perform inference
    results = model.predict(source=resized_image, conf=0.5, iou=0.5)

    # Draw bounding boxes on the original image
    resized_image = draw_bounding_box(results, resized_image)

    # Display the image with bounding boxes
    cv2.imshow('Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # webcam_main()
    picture_main("./src/test.jpg")
