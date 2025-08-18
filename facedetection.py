import cv2
from ultralytics import YOLO

# ---------- Detect camera index ----------
def get_external_camera_index(max_index=5):
    working_cams = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            working_cams.append(i)
            cap.release()
    if len(working_cams) > 1:
        return working_cams[-1]  # Likely external
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
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

# ---------- Load YOLOv8 model ----------
model = YOLO("yolov8n.pt")  # Replace with face-specific model if available

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, stream=True, conf=0.5)

    detections = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Class 0 is "person" in COCO dataset
                detections += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display number of detected objects
    text = f"Number of people detected = {detections}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
