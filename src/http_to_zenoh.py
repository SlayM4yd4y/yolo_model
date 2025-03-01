import cv2, zenoh, threading, time

class CameraStreamer:
    def __init__(self, session):
        self.session = session
        self.publisher = session.declare_publisher("camera/frame")
        self.lock = threading.Lock()
        self.camera_ip = None  
        self.cap = None
        self.running = True
        self.ip_received = threading.Event() 

    def update_camera(self, new_ip):
        with self.lock:
            if self.cap:
                self.cap.release()
            if new_ip:
                print(f"[ZENOH:CAMERA] 🔄 Updating camera IP to {new_ip}")
                self.camera_ip = new_ip
                self.cap = cv2.VideoCapture(self.camera_ip)
                self.ip_received.set() 

    def stream_frames(self):
        print("[ZENOH:CAMERA] Waiting for camera IP...")
        self.ip_received.wait() 

        print("[ZENOH:CAMERA] Starting frame streaming...")
        while self.running:
            with self.lock:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        _, buffer = cv2.imencode(".jpg", frame)
                        self.publisher.put(buffer.tobytes())
            time.sleep(0.03)

    def stop(self):
        with self.lock:
            self.running = False
            if self.cap:
                self.cap.release()

def on_ip_update(sample, camera_streamer):
    new_ip = bytes(sample.payload).decode("utf-8")
    if new_ip:
        camera_streamer.update_camera(new_ip)

def main():
    print("[ZENOH:CAMERA] Initializing Zenoh session...")
    session = zenoh.open(zenoh.Config())
    camera_streamer = CameraStreamer(session)
    session.declare_subscriber("camera/ip", lambda sample: on_ip_update(sample, camera_streamer))

    try:
        camera_streamer.stream_frames()
    except KeyboardInterrupt:
        print("[ZENOH:CAMERA] Exiting...")
    finally:
        camera_streamer.stop()
        session.close()

if __name__ == "__main__":
    main()
