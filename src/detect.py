import cv2
import torch
import zenoh
import numpy as np
from ultralytics import YOLO

model = YOLO("models/.pt") #TODO: sajat model

session = zenoh.open()
sub = session.declare_subscriber("video/frame")

def detect_objects(image):
    results = model(image)  
    return results

while True:
    data = sub.recv()
    np_arr = np.frombuffer(data.payload, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    results = detect_objects(frame)  #
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    session.put("video/detected", cv2.imencode(".jpg", frame)[1].tobytes())
