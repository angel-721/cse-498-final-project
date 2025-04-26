import torch
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

classes = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd', 'Qh', 'Qs']

model = YOLO("./models/yolov11n-cards.pt")
#test on single image and display results
# Load an image
# img = cv2.imread("./datasets/cards/test/images/001761433_jpg.rf.1a4a563beb6d588841bd8bd3a50c083b.jpg")
# img = cv2.imread("./test.jpg")
# # Resize the image to the input size of the model
# img = cv2.resize(img, (640, 640))

# # Perform inference
# results = model.predict(source=img, conf=0.5, iou=0.5)

# # Display results
# for result in results:
#         # Get the bounding boxes and labels
#         boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
#         scores = result.boxes.conf.cpu().numpy()  # Confidence scores
#         class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

#         # Draw bounding boxes on the image
#         for box, score, class_id in zip(boxes, scores, class_ids):
#             x1, y1, x2, y2 = box.astype(int)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(img, f'Class: {classes[int(class_id)]}, Score: {score:.2f}', (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    # Write the frame to the output file
    out.write(frame)

    # Resize the frame to the input size of the model
    resized_frame = cv2.resize(frame, (640, 640))
    # Perform inference
    results = model.predict(source=resized_frame, conf=0.5, iou=0.5)
    # Display results
    for result in results:
        # Get the bounding boxes and labels
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        # Draw bounding boxes on the image
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Class: {classes[int(class_id)]}, Score: {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
