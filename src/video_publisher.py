import cv2
import zenoh
import numpy as np
import time

session = zenoh.open()
pub = session.declare_publisher("video/frame")

cap = cv2.VideoCapture(0)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    _, buffer = cv2.imencode(".jpg", frame)
    pub.put(buffer.tobytes())  
    
    time.sleep(0.03)  

cap.release()
session.close()
