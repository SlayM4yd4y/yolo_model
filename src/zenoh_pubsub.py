import zenoh, json, cv2
import numpy as np

frame_p = "latest_frame.jpg"
frameCount = 1

def parse_detection_results(detected_ids):
    class_map = {
    0: "Aeroplane", 1: "Bicycle", 2: "Bird", 3: "Boat", 4: "Bottle",
    5: "Bus", 6: "Car", 7: "Cat", 8: "Chair", 9: "Cow",
    10: "Dining Table", 11: "Dog", 12: "Horse", 13: "Motorbike", 14: "Person",
    15: "Potted Plant", 16: "Sheep", 17: "Sofa", 18: "Train", 19: "TV Monitor",
    20: "Alkalmazotti Kártya", 21: "Hallgatói Kártya"}

    return [class_map.get(obj_id, "Unknown Class") for obj_id in detected_ids]


def callback(sample):
    global frameCount
    try:
        if sample.key_expr == "camera/frame":
            frame_bytes = bytes(sample.payload)
            
            if not frame_bytes:
                print("[ZENOH:SUB] Received empty frame data.")
                return

            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("[ZENOH:SUB] Failed to decode frame.")
                return

            cv2.imwrite(frame_p, frame)
            print("[ZENOH:SUB] New camera frame saved.")

        else:
            payload = bytes(sample.payload).decode("utf-8")
            
            try:
                data = json.loads(payload)
                if "detected_objects" in data:
                    detected_ids = data["detected_objects"] 
                    detected_names = parse_detection_results(detected_ids)
                    if detected_names == []:
                        print("[ZENOH:SUB] No objects detected.")
                        frameCount += 1
                    else:
                        print(f"[ZENOH:SUB] Detection results at frame number {frameCount} : {", ".join(detected_names)}")
                        frameCount += 1
                elif "training_status" in data:
                    print(f"[ZENOH:SUB] Training status: {data['training_status']}")
                else:
                    print("[ZENOH:SUB] Unknown JSON format.")
            except json.JSONDecodeError:
                print(f"[ZENOH:SUB] Raw message: {payload}")

    except Exception as e:
        print(f"[ZENOH:SUB] Error in callback: {e}")

try:
    session = zenoh.open(zenoh.Config())
    sub1 = session.declare_subscriber("yolo/**", callback)
    sub2 = session.declare_subscriber("camera/frame", callback)
    print("[ZENOH:SUB] Zenoh subscriber started. Listening...")
    print("[ZENOH:SUB] Press Enter to stop listening...")
    input()
except Exception as e:
    print(f"[ZENOH:SUB] Fatal error: {e}")

finally:
    print("[ZENOH:SUB] Closing Zenoh session...")
    sub1.undeclare()
    sub2.undeclare()
    session.close()
    print("[ZENOH:SUB] Subscriber stopped.")
