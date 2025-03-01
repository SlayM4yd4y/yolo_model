import os, cv2, argparse, torch, zenoh, json, time, threading, signal, sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO

session = None

class ZenohCameraSubscriber:
    def __init__(self, session, topic="camera/frame"):
        self.frame = None
        self.lock = threading.Lock()
        self.subscriber = session.declare_subscriber(topic, self.callback)

    def callback(self, sample):
        np_arr = np.frombuffer(bytes(sample.payload), np.uint8)
        if np_arr is not None and len(np_arr) > 0:
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            print("[ZENOH:SUBSCRIBER] Empty or corrupted frame!")
            return
        with self.lock:
            self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

def signal_handler(sig, frame):
    global session
    if session:
        print(f"[ZENOH:DETECT] Termination signal received. Exiting...")
        session.close()
    sys.exit(0)

def get_latest_prediction_folder(base_path):
    latest_folder = None
    latest_time = 0

    for entry in Path(base_path).iterdir():
        if entry.is_dir() and entry.name.startswith("predict"):
            folder_time = entry.stat().st_mtime
            if folder_time > latest_time:
                latest_time = folder_time
                latest_folder = entry

    return str(latest_folder) if latest_folder else None


def run_detection(weights, source, save_results, save_dir, show, publisher):
    print("[YOLOv11:DETECT] Starting detection...")

    res = weights.predict(source=source, save=save_results, project=save_dir, show=show, save_txt=True, stream=True)
    for r in res:
        detected_objects = [int(box.cls[0]) for box in r.boxes]  
        object_count = len(detected_objects)

        detection_message = json.dumps({
            "object_count": object_count,
            "detected_objects": detected_objects
        })
        publisher.put(detection_message) 

    latest_folder = get_latest_prediction_folder(save_dir)

    if latest_folder:
        print(f"Latest prediction folder: {latest_folder}")
        detected_objects = parse_detection_results(Path(latest_folder) / "labels")
        return detected_objects
    else:
        print("No predictions found.")
        return ["No objects detected"]

def parse_detection_results(results_dir):
    detected_objects = []
    
    class_map = {
        0: "Aeroplane", 1: "Bicycle", 2: "Bird", 3: "Boat", 4: "Bottle",
        5: "Bus", 6: "Car", 7: "Cat", 8: "Chair", 9: "Cow",
        10: "Dining Table", 11: "Dog", 12: "Horse", 13: "Motorbike", 14: "Person",
        15: "Potted Plant", 16: "Sheep", 17: "Sofa", 18: "Train", 19: "TV Monitor",
        20: "Alkalmazotti Kártya", 21: "Hallgatói Kártya"
    }

    if not results_dir.exists():
        print("Detection results folder does not exist.")
        return ["No objects detected"]

    for file in results_dir.iterdir():
        if file.suffix == ".txt":
            with file.open() as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        object_name = class_map.get(class_id, "Unknown Class")
                        detected_objects.append(object_name)

    if not detected_objects:
        detected_objects.append("No objects detected")

    return detected_objects

def detect_and_publish(config, weights, source_type, source, camera_ip, save_results, save_dir, show):
    print("[ZENOH:DETECT] Initializing Zenoh session...")
    global session
    zenoh_config = zenoh.Config.from_file(config) if config else zenoh.Config()
    session = zenoh.open(zenoh_config)
    publisher = session.declare_publisher("yolo/detection/results")
    ip_cam_pub = session.declare_publisher("camera/ip")
    model = YOLO(weights)

    if camera_ip:
        print(f"[ZENOH:DETECT] Publishing camera IP address: {camera_ip}")
        ip_cam_pub.put(camera_ip.encode("utf-8"))

    camera = None
    if source_type == "camera":
        camera = ZenohCameraSubscriber(session, "camera/frame")
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        max_retries, retry_count = 50, 0
        while True:
            if source_type == "camera":
                frame = camera.get_frame()
                if frame is None:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print("[ZENOH:DETECT] No frames received. Exiting...")
                        break
                    time.sleep(0.1)
                    continue
                retry_count = 0
                run_detection(model, frame, save_results, save_dir, show, publisher)
            else:
                run_detection(model, source, save_results, save_dir, show, publisher)

            if source_type in ["image", "video"]:
                break
            if cv2.waitKey(1) & 0xFF == 26:  #
                print("[ZENOH:DETECT] Ctrl+Z pressed. Closing OpenCV window...")
                cv2.destroyAllWindows()
                break
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("[ZENOH:DETECT] Stopping detection...")
    finally:
        if session:
            session.close()
        print("[ZENOH:DETECT] Zenoh session closed.")
    

def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Detection with Zenoh")
    parser.add_argument("--config", type=str, default="", help="Path to the Zenoh configuration file")
    parser.add_argument("--weights", type=str, required=True, help="Path to the YOLOv11 model weights")
    parser.add_argument("--source_type", type=str, choices=["image", "video", "camera"], required=True, help="Type of source (image, video, or camera)")
    parser.add_argument("--video_path", type=str, default="", help="Path to the video file")
    parser.add_argument("--image_path", type=str, default="", help="Path to the image file")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--camera_ip", type=str, default="", help="IP address of the network camera")
    parser.add_argument("--save_results", action="store_true", help="Save detection results")
    parser.add_argument("--save_dir", type=str, default="det_results", help="Directory to save detection results")
    parser.add_argument("--show", action="store_true", help="Display image results")
    args = parser.parse_args()

    if args.source_type == "video":
        if not args.video_path:
            print("[ERROR] Video source path is missing!")
            return
        source = args.video_path

    elif args.source_type == "image":
        if not args.image_path:
            print("[ERROR] Image source path is missing!")
            return
        source = args.image_path
    elif args.source_type == "camera":
        source = None  
    else:
        print(f"[ZENOH:DETECT] Invalid source type. Exiting...")
        return

    detect_and_publish(args.config, args.weights, args.source_type, source, args.camera_ip, args.save_results, args.save_dir, args.show)
 

if __name__ == "__main__":
    main()
