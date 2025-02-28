import zenoh, json, cv2
import numpy as np

frame_p = "latest_frame.jpg"

def callback(sample):
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
            print(f"[ZENOH:SUB] Received data on {sample.key_expr}: {payload}")
            try:
                data = json.loads(payload)
                if "detected_objects" in data:
                    print(f"[ZENOH:SUB] Detection results: {data['detected_objects']}")
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
