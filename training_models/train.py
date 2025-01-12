import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

red_lower = np.array([170, 100, 100])
red_upper = np.array([180, 255, 255])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
green_lower = np.array([70, 100, 100])
green_upper = np.array([80, 255, 255])

def detect_vehicles_yolo(frame):
    """
    Detects vehicles in an image using YOLOv8.

    Args:
        frame: The input image (BGR format).

    Returns:
        frame: The image with detected vehicles marked (bounding boxes).
        vehicles: List of (x, y, w, h) coordinates for detected vehicles.
    """
    results = model(frame)

    vehicles = []
    for result in results[0].boxes:  
        x1, y1, x2, y2 = result.xyxy[0] 
        conf = result.conf[0]  
        cls = result.cls[0]  

        if conf > 0.5 and int(cls) == 2: 
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            vehicles.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  #

    return frame, vehicles


def track_vehicles(frame, previous_centroids):
    """
    Tracks vehicles in the current frame based on their centroids.
    Args:
        frame: The current frame (BGR format).
        previous_centroids: List of centroids from the previous frame.

    Returns:
        frame: The image with tracking lines and vehicle centroids.
        current_centroids: Updated list of centroids.
    """
    frame, vehicles = detect_vehicles_yolo(frame)

    current_centroids = [(x + w // 2, y + h // 2) for (x, y, w, h) in vehicles]

    if previous_centroids:
        for prev_centroid in previous_centroids:
            closest_vehicle = None
            min_distance = float('inf')

            for centroid in current_centroids:
                distance = np.linalg.norm(np.array(prev_centroid) - np.array(centroid))
                if distance < min_distance:
                    min_distance = distance
                    closest_vehicle = centroid

            if closest_vehicle:
                cv2.line(frame, prev_centroid, closest_vehicle, (255, 0, 0), 2)  

    return frame, current_centroids

cap = cv2.VideoCapture('ee.mp4')  
previous_centroids = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, current_centroids = track_vehicles(frame, previous_centroids)

    previous_centroids = current_centroids

    resized_img = cv2.resize(frame, (1024, 768))  

    cv2.imshow("Vehicle Detection and Tracking", resized_img)

    print(f"Detected vehicles: {len(current_centroids)}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
