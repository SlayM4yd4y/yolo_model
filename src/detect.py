import os, cv2, argparse, torch, zenoh, json, time, subprocess
from pathlib import Path
from ultralytics import YOLO

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


def run_detection(weights, source, save_results, save_dir, show):
    print("[YOLOv11:DETECT] Starting detection...")

    model = YOLO(weights)  
    model.predict(source=source, save=save_results, project=save_dir, show=show, save_txt=True)

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

def detect_and_publish(config, weights, source, save_results, save_dir, show):
    try:
        print("[ZENOH:DETECT] Initializing Zenoh session...")
        if config: 
            zenoh_config = zenoh.Config.from_file(config)
        else:
            zenoh_config = zenoh.Config()
        session = zenoh.open(zenoh_config)
        publisher = session.declare_publisher("yolo/detection/results")
        detected_objects = run_detection(weights, source, save_results, save_dir, show)
        detection_message = json.dumps({"detected_objects": detected_objects})
        publisher.put(detection_message)
        print(f"[ZENOH:DETECT] Published detection results: {detection_message}")
        session.close()
    except Exception as e:
        print(f"[ZENOH:DETECT] Error during detection and publishing: {e}")

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

    cam_proc = None
    if args.source_type == "video":
        source = args.video_path
    elif args.source_type == "image":
        source = args.image_path
    elif args.source_type == "camera":
        source = "latest_frame.jpg"
        print("[YOLOv11:DETECT] Starting camera stream...")
        cam_cmd = ["python", "src/camera_pub.py", "--camera_id", str(args.camera_id), "--camera_ip", args.camera_ip]
        cam_proc = subprocess.Popen(cam_cmd) 
    else:
        print(f"[ZENOH:DETECT] Invalid source type. Exiting...")
        return

    detect_and_publish(args.config, args.weights, source, args.save_results, args.save_dir, args.show)
    if cam_proc is not None:
        cam_proc.terminate()
        print(f"[ZENOH:DETECT] Camera publisher stopped.")

if __name__ == "__main__":
    main()
