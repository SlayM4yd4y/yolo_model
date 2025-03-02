## YOLOv11 Zenoh

**ZENOH YOLO MODEL** - Zenoh-based Object Detection

A deep learning-based YOLOv11 object detection project utilizing Zenoh protocol for efficient communication and data streaming. This implementation supports real-time detection from multiple input sources, including cameras, video files, and images.Developed and tested in a WSL (Windows Subsystem for Linux) environment. Written in Python and C++ for a university class.

### Features

- ✅ Live Camera Detection: Process camera frames in real time via Zenoh.
- ✅ Video & Image Detection: Detect objects in video files or images.
- ✅ Multi-class Support: Recognizes 22 object classes, including PASCAL VOC 2012 dataset objects and custom classes like student and employee ID cards.
- ✅ Zenoh Messaging System: Sends detection results via Zenoh topics, making it suitable for distributed and edge AI applications.

### Clone the repository
``` 
git clone https://github.com/SlayM4yd4y/zenoh_yolo_model.git
cd zenoh_yolo_model
pip install -r requirements.txt 
```
> You will also need:

**YOLOv11**
```
git clone https://github.com/ultralytics/ultralytics.git 
cd ultralytics
pip install -r requirements.txt  
```
**Zenoh**
```
git clone https://github.com/eclipse-zenoh/zenoh.git
cd zenoh-python
```
## Run 

<div align="center"><h3>For more details visit the /wiki of this repository.</h3></div>

## Diagram
``` mermaid
graph LR;

train([ train.py]):::red --> pubsub[ zenoh_pubsub.py]:::light
detector([ detector.py]):::red --> pubsub
detector --> conv([ http_to_zenoh.py]):::red --> cp([ camera_pub.py]):::red --> pubsub
cg([ card_gen.cpp]):::light
ca([ card_augmenter.cpp]):::light
ic([ identifier_cleanup]):::light
gpp([ get_package_path.cpp]):::light

classDef light fill:#34aec5,stroke:#152742,stroke-width:2px,color:#152742  
classDef dark fill:#152742,stroke:#34aec5,stroke-width:2px,color:#34aec5
classDef white fill:#ffffff,stroke:#152742,stroke-width:2px,color:#152742
classDef red fill:#ef4638,stroke:#152742,stroke-width:2px,color:#fff
```