import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')

# HSV color thresholds for traffic light detection
red_lower = np.array([170, 100, 100])
red_upper = np.array([180, 255, 255])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
green_lower = np.array([70, 100, 100])
green_upper = np.array([80, 255, 255])

def detect_traffic_light(frame):
    """
    Detects the traffic light's current color.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask for red, yellow, and green
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    if cv2.countNonZero(red_mask) > 500:
        return "Red"
    elif cv2.countNonZero(yellow_mask) > 500:
        return "Yellow"
    elif cv2.countNonZero(green_mask) > 500:
        return "Green"
    return "Unknown"

def detect_vehicles_yolo(frame):
    """
    Detects vehicles using YOLOv8.
    """
    results = model(frame)
    vehicles = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        conf = result.conf[0]
        cls = result.cls[0]

        if conf > 0.5 and int(cls) == 2:  # Class 2 is typically "car" in YOLO
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            vehicles.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, vehicles

def traffic_light_logic(vehicles, light_status, ev_distance, tg_qd_threshold=10):
    """
    Implements the traffic light algorithm for Emergency Vehicle Priority.
    """
    ns_r = len([v for v in vehicles if v[1] > 400])  # Vehicles near the light
    tg_qd = ns_r * 2  # Example delay calculation

    if light_status == "Red" and ev_distance <= 50:
        if tg_qd < tg_qd_threshold:
            light_status = "Green"
    elif light_status == "Green" and ev_distance > -20:
        pass  # Stay green while EV is near
    else:
        light_status = "Red"

    return light_status

# Video capture
cap = cv2.VideoCapture('ee.mp4')
previous_centroids = None
light_status = "Red"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect vehicles
    frame, vehicles = detect_vehicles_yolo(frame)

    # Detect traffic light status
    light_status = detect_traffic_light(frame)

    # Simulate EV distance (example: moving closer over time)
    ev_distance = np.random.randint(-50, 100)  # Example dynamic distance

    # Apply traffic light logic
    light_status = traffic_light_logic(vehicles, light_status, ev_distance)

    # Display light status and EV distance
    cv2.putText(frame, f"Light: {light_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"EV Distance: {ev_distance}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Resize for display
    resized_img = cv2.resize(frame, (1024, 768))
    cv2.imshow("Vehicle Detection and Traffic Light Logic", resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
