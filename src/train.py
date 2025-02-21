import argparse, os, yaml, torch
from ultralytics import YOLO

def get_next_train_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith("train") and d[5:].isdigit()]
    train_nums = [int(d[5:]) for d in existing_dirs if d[5:].isdigit()]
    next_train_num = max(train_nums) + 1 if train_nums else 1
    return os.path.join(base_dir, f"train{next_train_num}")

def modify_model_config(model_config, scale):
    with open(model_config, "r") as f:
        model_params = yaml.safe_load(f)
    if "scales" in model_params and scale in model_params["scales"]:
        width, depth, img_size = model_params["scales"][scale]
        model_params["width_multiple"] = width
        model_params["depth_multiple"] = depth
        model_params["imgsz"] = int(img_size)  
        print(f"🔹 Using scale '{scale}': width={width}, depth={depth}, img_size={img_size}")
    else:
        print(f"Scale '{scale}' not found in model config! Using default values.")

    temp_model_yaml = "temp_yolov11.yaml"
    with open(temp_model_yaml, "w") as f:
        yaml.safe_dump(model_params, f)

    return temp_model_yaml, int(img_size)

def train_yolo(model_config, data_config, hyp_config, epochs, batch_size, img_size, output_format, save_dir):
    train_folder = get_next_train_folder(save_dir)
    os.makedirs(train_folder, exist_ok=True)

    #modified_model_config, img_size = modify_model_config(model_config, scale)

    print(f"🔹 Training YOLOv11 with:")
    print(f"- Model config: {model_config}")
    #print(f"- Scale: {scale}")
    print(f"- Data config: {data_config}")
    print(f"- Hyperparameters: {hyp_config}")
    print(f"- Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    print(f"- Saving to: {train_folder}")

    hyp_params = {}
    if os.path.exists(hyp_config):
        with open(hyp_config, "r") as f:
            hyp_params = yaml.safe_load(f)
            print(f"✅ Loaded hyperparameters from {hyp_config}")

    model = YOLO(model_config)
    model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device="cuda",
        project=train_folder,
        name="",
        **hyp_params  
    )

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(train_folder, "yolov11.pt")

    if output_format == "onnx":
        output_path = os.path.join(train_folder, "yolov11.onnx")
        model.export(format="onnx", path=output_path)
        print(f"✅ Model exported as ONNX: {output_path}")
    else:
        torch.save(model.model.state_dict(), output_path)  
        print(f"✅ Model trained and saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11")

    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv11 model YAML (e.g., yolov11.yaml)")
    """parser.add_argument("--scale", type=str, choices=["n", "s", "m", "l", "x"], default="s",
                    help="Model scale (n, s, m, l, x). Default: s")"""
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML (e.g., data.yaml)")
    parser.add_argument("--hyp", type=str, default="hyp.yaml", help="Path to hyperparameters YAML (default: hyp.yaml)")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640)")
    parser.add_argument("--format", type=str, choices=["pt", "onnx"], default="pt", help="Model output format (default: pt)")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save the trained model (default: models/)")

    args = parser.parse_args()
    train_yolo(args.model, args.data, args.hyp, args.epochs, args.batch, args.imgsz, args.format, args.save_dir)
