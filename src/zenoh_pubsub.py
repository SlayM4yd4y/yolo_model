import cv2
import zenoh
import numpy as np

session = zenoh.open()
sub = session.declare_subscriber("video/detected")

while True:
    data = sub.recv()
    np_arr = np.frombuffer(data.payload, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    cv2.imshow("Detected Objects", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
