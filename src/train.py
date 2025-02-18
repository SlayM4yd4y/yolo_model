import ultralytics
from ultralytics import YOLO

model = YOLO("yolov11.yaml")  
data_path = ""  #TODO: sajat adatok

model.train(data=data_path, epochs=100, imgsz=640)
model.export(format="onnx")  
