import argparse, os, yaml, torch, zenoh, json5
from ultralytics import YOLO
from zenoh import Session

def get_next_train_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith("train") and d[5:].isdigit()]
    train_nums = [int(d[5:]) for d in existing_dirs if d[5:].isdigit()]
    next_train_num = max(train_nums) + 1 if train_nums else 1
    return os.path.join(base_dir, f"train{next_train_num}")

def train_yolo(model_config, data_config, hyp_config, epochs, batch_size, img_size, output_format, save_dir, optimizer):
    #config = zenoh.Config.from_file('config/zenoh_trainconfig.json5')
    sessionZ = zenoh.open(zenoh.Config())
    
    train_folder = get_next_train_folder(save_dir)
    os.makedirs(train_folder, exist_ok=True)

    print(f"[YOLOv11:TRAIN] Training YOLOv11 with:")
    print(f"- Model config: {model_config}")
    print(f"- Data config: {data_config}")
    print(f"- Hyperparameters: {hyp_config}")
    print(f"- Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    print(f"- Saving to: {train_folder}")
    print(f"- Optimizer: {optimizer}")

    sessionZ.put("yolo/trainig/status", "Training started...")
    hyp_params = {}
    if os.path.exists(hyp_config):
        with open(hyp_config, "r") as f:
            hyp_params = yaml.safe_load(f)
            print(f"[YOLOv11:TRAIN] Loaded hyperparameters from {hyp_config}")

    model = YOLO(model_config)
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device="cuda",
        project=train_folder,
        name="results",
        optimizer=optimizer,
        **hyp_params  
    )
    
    loss = results.speed.get("loss", "unknown") 
    training_info = (f"Epoch number: {epochs}, batches: {batch_size}\n"
                 f"Loss: {loss}, data: {data_config},\n"
                 f"Model: {model_config}, hyp: {hyp_config}")

    sessionZ.put("yolo/training/info", training_info)

    weights_dir = os.path.join(train_folder, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    pt_path = os.path.join(weights_dir, "yolov11n.pt")
    torch.save(model.model.state_dict(), pt_path) #csak a sulyokat mentem, resultsban ugyis benne a teljes best.pt
    print(f"[YOLOv11:TRAIN] Model trained and saved as PT: {pt_path}")
    sessionZ.put("yolo/training/model_path", pt_path)
    if output_format == "onnx":
        model.export(format="onnx")
        print(f"[YOLOv11:TRAIN] Model exported as ONNX.")
        sessionZ.put("yolo/training/model_format", "onnx")
    sessionZ.put("yolo/training/status", "Completed!")
    sessionZ.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11")

    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv11 model YAML (e.g., yolov11.yaml)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML (e.g., data.yaml)")
    parser.add_argument("--hyp", type=str, default="hyp.yaml", help="Path to hyperparameters YAML (default: hyp.yaml)")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640)")
    parser.add_argument("--format", type=str, choices=["pt", "onnx"], default="pt", help="Model output format (default: pt)")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save the trained model (default: models/)")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer to use for training (AdamW,SGD,etc.) ")

    args = parser.parse_args()
    train_yolo(args.model, args.data, args.hyp, args.epochs, args.batch, args.imgsz, args.format, args.save_dir, args.optimizer)
