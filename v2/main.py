import cv2
import numpy as np
from ultralytics import YOLO

# === Change only this ===
VIDEO_PATH = '/Users/pradeeppatil/workspace/ra-work/v2/videos/acre2.mp4'

# Load fast YOLO model
model = YOLO("yolov8n.pt")  # smallest model
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Speed settings
DOWNSCALE = 1  
YOLO_IMGSZ = 256  # YOLO input image size
SKIP_FRAMES = 5   # Process every Nth frame

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("Failed to read video.")
    exit()

# Downsample first frame
prev_small = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        continue  # skip this frame

    # Downsample current frame
    small = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
    curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # YOLO detection on smaller frame
    results = model(small, imgsz=YOLO_IMGSZ, verbose=False)[0]
    boxes = results.boxes

    for box in boxes:
        cls_id = int(box.cls)
        if cls_id not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        p0 = np.array([[[cx, cy]]], dtype=np.float32)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

        if st[0][0] == 1:
            dx = p1[0][0][0] - p0[0][0][0]
            if dx > 0.5:
                scale = int(1 / DOWNSCALE)
                x1f, y1f, x2f, y2f = x1 * scale, y1 * scale, x2 * scale, y2 * scale
                cv2.rectangle(frame, (x1f, y1f), (x2f, y2f), (0, 255, 0), 2)
                cv2.putText(frame, f"Right [{model.names[cls_id]}]", (x1f, y1f - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Vehicles Moving Right", frame)
    if cv2.waitKey(1) == 27:
        break

    prev_gray = curr_gray.copy()

cap.release()
cv2.destroyAllWindows()
