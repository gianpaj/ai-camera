import cv2
import time
# import io
# import asyncio
from pprint import pprint
# from picamera2 import Picamera2
# from picamera2.encoders import MJPEGEncoder, Quality
# from picamera2.outputs import FileOutput
# from fastapi import FastAPI, WebSocket
# from threading import Condition
# from contextlib import asynccontextmanager
from ultralytics import YOLO
# import numpy as np

# Load an official or custom model
model = YOLO("models/yolov8m.pt")
# CoreML model
# model = YOLO("models/YOLOv8-CoreML/yolov8m.mlpackage")

# https://docs.ultralytics.com/modes/predict/#inference-arguments
confidence_threshold = 0.25
intersection_over_union = 0.7

# export the images of the individual steps of the detection process
visualize_debug = False
# only detect the specified classes
# 0 = person, 1 = bicycle, 2 = car, 3 = motorcycle, 5 = bus, 7 = truck,
# 14 = bird, 15 = cat, 16 = dog, 17 = horse, etc.
# predictions_class_ids = [0, 1, 2, 3, 5, 7, 14, 15, 16, 17]
predictions_class_ids = None

def capture_image():
    # Initialize the camera
    camera = cv2.VideoCapture(1)
    
    # Check if the camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Allow the camera to warm up
    # time.sleep(2)
    
    # Capture a frame
    ret, frame = camera.read()
    
    if ret:
        # Generate a filename with timestamp
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        timestamp = 321321
        original_filename = f"original_image_{timestamp}.jpg"
        annotated_filename = f"annotated_image_{timestamp}.jpg"

        # Perform object detection
        results = model(frame, conf=confidence_threshold, iou=intersection_over_union, 
                        half=False, max_det=10, visualize=visualize_debug, classes=predictions_class_ids)

        # Process results list
        for i, result in enumerate(results):
            # print(f"Batch {i}: {result}")
            boxes = result.boxes  # Boxes object containing the detection bounding boxes.
            masks = result.masks  # Masks object containing the detection masks.
            keypoints = result.keypoints  # Keypoints object containing detected keypoints for each object.
            probs = result.probs  # Probs object containing probabilities of each class for classification task.
            obb = result.obb  # OBB object containing oriented bounding boxes.
            speed = result.speed  # dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.
            # names = result.names  # A dictionary of class names.
            pprint({
                "boxes": boxes,
                "masks": masks,
                "keypoints": keypoints,
                "probs": probs,
                "obb": obb,
                "speed": speed,
                # "names": names
            })
            # result.show()  # display to screen
            # Save the annotated frame as an image file
            result.save(filename=f"{annotated_filename}")
            print(f"Image saved as {annotated_filename}")

        # _, annotated_frame_jpeg = cv2.imencode('.jpg', annotated_frame)

        
        # Save the original captured frame as an image file
        cv2.imwrite(original_filename, frame)
        print(f"Image saved as {original_filename}")

    else:
        print("Error: Failed to capture image.")
    
    # Release the camera
    camera.release()

if __name__ == "__main__":
    capture_image()