import argparse
import os
from ultralytics import YOLO

def detect_video(model_path, video_path, save_dir, view_img):
    os.makedirs(save_dir, exist_ok=True)

    model = YOLO(model_path)
    model.predict(source=video_path, save=True, project=save_dir, name="detections", show=view_img)

    print(f"✅ Detection completed! Results saved in {save_dir}/detections")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv11 detection on a video.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model (e.g., yolov11.pt)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save detection results (default: results/)")
    parser.add_argument("--view-img", action="store_true", help="Show detection results in a window")

    args = parser.parse_args()
    detect_video(args.model, args.video, args.save_dir, args.view_img)

