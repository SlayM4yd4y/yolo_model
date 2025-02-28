import cv2, zenoh, time, argparse, json

def main(camera_id, camera_ip, fps):
    session = zenoh.open(zenoh.Config())
    publisher = session.declare_publisher("camera/frame")

    cap = cv2.VideoCapture(camera_ip if camera_ip else camera_id)
    if not cap.isOpened():
        print(f"[CAMERA_PUB] Failed to open camera (ID: {camera_id}, IP: {camera_ip})")
        return

    print(f"[CAMERA_PUB] Streaming from {'IP camera' if camera_ip else 'webcam'}...")

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[CAMERA_PUB] Failed to capture frame.")
            break

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        publisher.put(frame_bytes)

        elapsed_time = time.time() - start_time
        sleep_time = max(1.0 / fps - elapsed_time, 0)
        time.sleep(sleep_time)

    cap.release()
    session.close()
    print("[CAMERA_PUB] Stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Publisher for YOLOv11")
    parser.add_argument("--camera_id", type=int, default=0, help="Local camera ID")
    parser.add_argument("--camera_ip", type=str, default="", help="IP camera address")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    main(args.camera_id, args.camera_ip, args.fps)
