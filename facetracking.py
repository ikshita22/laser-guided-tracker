import cv2
from ultralytics import YOLO
import pyfirmata
import numpy as np

# ---------- Detect camera index ----------
def get_external_camera_index(max_index=5):
    working_cams = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            working_cams.append(i)
            cap.release()
    if len(working_cams) > 1:
        return working_cams[-1]
    elif working_cams:
        return working_cams[0]
    else:
        return None

cam_index = get_external_camera_index()
if cam_index is None:
    print("No camera found!")
    exit()
print(f"Using camera index: {cam_index}")

# ---------- Camera setup ----------
ws, hs = 1280, 720
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(3, ws)
cap.set(4, hs)

# ---------- Arduino setup ----------
port = "COM7"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s')
servo_pinY = board.get_pin('d:10:s')

# ---------- YOLOv8 Model ----------
model = YOLO("yolov8n.pt")  # You can replace with a face-trained model if available
servoPos = [90, 90]

while True:
    success, img = cap.read()
    if not success:
        break

    results = model.predict(img, stream=True, conf=0.5)

    detected = False
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # class 0 is 'person' in COCO dataset
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                fx = (x1 + x2) // 2
                fy = (y1 + y2) // 2

                servoX = np.interp(fx, [0, ws], [180, 0])
                servoY = np.interp(fy, [0, hs], [0, 180])
                servoPos[0] = np.clip(servoX, 0, 180)
                servoPos[1] = np.clip(servoY, 0, 180)

                ch_size = 15
                cv2.line(img, (fx - ch_size, fy), (fx + ch_size, fy), (0, 0, 0), 2)
                cv2.line(img, (fx, fy - ch_size), (fx, fy + ch_size), (0, 0, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                detected = True
                break
        if detected:
            break

    if not detected:
        servoPos = [90, 90]
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    servo_pinX.write(servoPos[0])
    servo_pinY.write(servoPos[1])

    cv2.imshow("YOLO Face Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
