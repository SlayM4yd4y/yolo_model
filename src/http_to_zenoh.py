import cv2, zenoh, time

session = zenoh.open(zenoh.Config())
publisher = session.declare_publisher("camera/frame")

camera_url = "IP_CAMERA_URL"
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("[ERROR] Camera cannot be opened!")
    exit(1)

print("[ZENOH:PUBLISHER] Camera stream started...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Couldn't read frame!")
            time.sleep(0.1)
            continue
       
        _, buffer = cv2.imencode(".jpg", frame)
        publisher.put(buffer.tobytes())

        time.sleep(0.03)  
except KeyboardInterrupt:
    print("[ZENOH:PUBLISHER] Exiting...")
finally:
    cap.release()
    session.close()
