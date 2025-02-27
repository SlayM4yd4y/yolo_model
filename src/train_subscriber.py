import zenoh

def callback(sample):
    try:
        payload = bytes(sample.payload).decode("utf-8")
        print(f"Received data: {sample.key_expr}: {payload}")
    except Exception as e:
        print(f"Error in callback: {e}")

session = zenoh.open(zenoh.Config())
key_expr = "yolo/training/**"  
sub = session.declare_subscriber(key_expr, callback)

print("Zenoh subscriber started. Listening...")
print("Press Enter to stop listening...")
input() 
sub.undeclare()
session.close()
print(">>>Subscriber stopped.")
