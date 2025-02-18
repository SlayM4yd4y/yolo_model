import argparse
import os
from ultralytics import YOLO

def train_yolo(model_config, data_config, hyp_config, epochs, batch_size, img_size, output_format, save_dir):
    print(f"🔹 Training YOLOv11 with:")
    print(f"- Model config: {model_config}")
    print(f"- Data config: {data_config}")
    print(f"- Hyperparameters: {hyp_config}")
    print(f"- Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")

    model = YOLO(model_config)
    model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        hyp=hyp_config,
        device="cuda"
    )
    os.makedirs(save_dir, exist_ok=True)

    if output_format == "onnx":
        output_path = os.path.join(save_dir, "yolov11.onnx")
        model.export(format="onnx", path=output_path)
        print(f"✅ Model exported as ONNX: {output_path}")
    else:
        output_path = os.path.join(save_dir, "yolov11.pt")
        model.save(output_path)
        print(f"✅ Model trained and saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11 from scratch with custom settings.")

    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv11 model YAML (e.g., yolov11.yaml)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML (e.g., data.yaml)")
    parser.add_argument("--hyp", type=str, default="hyp.yaml", help="Path to hyperparameters YAML (default: hyp.yaml)")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640)")
    parser.add_argument("--format", type=str, choices=["pt", "onnx"], default="pt", help="Model output format (default: pt)")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save the trained model (default: models/)")
    args = parser.parse_args()

    train_yolo(args.model, args.data, args.hyp, args.epochs, args.batch, args.imgsz, args.format, args.save_dir)
