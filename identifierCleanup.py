import os

DATASET_DIR = "/home/shin/projects/yolo_model/dataset"
SUBFOLDERS = ["train/images", "train/labels", "valid/images", "valid/labels"]

def cleanup_identifier_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".Identifier"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Törölve: {file_path}")
                except Exception as e:
                    print(f"Nem sikerült törölni: {file_path} - {e}")

if __name__ == "__main__":
    for subfolder in SUBFOLDERS:
        folder_path = os.path.join(DATASET_DIR, subfolder)
        if os.path.exists(folder_path):
            print(f"Ellenőrzés: {folder_path}")
            cleanup_identifier_files(folder_path)
        else:
            print(f"Nem található: {folder_path}")
