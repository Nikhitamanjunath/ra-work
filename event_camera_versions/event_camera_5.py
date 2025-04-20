import cv2
import numpy as np
import time
from statistics import mean

# Zoom‑out factor (e.g. 0.5 shows the frame at half size)
zoom_factor = 0.25

# Load the video
cap = cv2.VideoCapture("/Users/pradeeppatil/workspace/ra-work/acre.mp4")
ret, prev_frame = cap.read()
if not ret:
    print("Failed to read from video.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
height, width = prev_frame.shape[:2]

# Compute zoomed output size
out_w, out_h = int(width * zoom_factor), int(height * zoom_factor)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_event_zoomed.avi", fourcc, 30.0, (out_w, out_h))

detections = []
x_offset = 0
y_offset = 450
center_x, center_y = ((width // 2)+x_offset), ((height // 2)+y_offset)
cushion = 100
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(frame, 0.7, mask_bgr, 0.3, 0)

    # Draw center box
    cv2.rectangle(
        combined,
        (center_x - cushion, center_y - cushion),
        (center_x + cushion, center_y + cushion),
        (255, 0, 0), 2
    )
    cv2.circle(combined, (center_x, center_y), 5, (255, 255, 0), -1)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 150000:
            continue

        if area > 500000:
            detections.append("truck")
            continue


        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(combined, f"Area: {int(area)}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        box_cx, box_cy = x + w // 2, y + h // 2
        cv2.circle(combined, (box_cx, box_cy), 5, (0, 0, 255), -1)

        if abs(box_cx - center_x) <= cushion and abs(box_cy - center_y) <= cushion:
            ts = round(time.time() - start_time, 2)
            detections.append({"timestamp": ts, "width": w, "height": h, "area": area})
            print(detections[-1])

    prev_gray = gray.copy()

    # Resize for zoom‑out
    zoomed = cv2.resize(combined, (out_w, out_h), interpolation=cv2.INTER_AREA)
    out.write(zoomed)
    cv2.imshow("Edge Temporal Motion Detection (Zoomed Out)", zoomed)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

def group_and_average_detections(d, time_window=5.0):
    if not d:
        return []

    # Sort detections by timestamp
    d = sorted(d, key=lambda x: x['timestamp'])

    grouped = []
    current_group = [d[0]]
    prev_truck = False
    for i in range(1, len(d)):
        if d[i] == "truck" and prev_truck:
            continue
        if d[i] == "truck":
            current_group.append("truck")
            prev_truck = True
        else:
            prev_truck = False
        if d[i]['timestamp'] - current_group[0]['timestamp'] <= time_window:
            current_group.append(d[i])
        else:
            grouped.append(current_group)
            current_group = [d[i]]

    # Add the last group
    if current_group:
        grouped.append(current_group)

    # Now average each group
    averaged = []
    for group in grouped:
        avg_w = mean([item['width'] for item in group])
        avg_h = mean([item['height'] for item in group])
        avg_area = mean([item['area'] for item in group])
        mid_timestamp = group[len(group) // 2]['timestamp']

        averaged.append({
            'timestamp': round(mid_timestamp, 2),
            'width': round(avg_w, 1),
            'height': round(avg_h, 1),
            'area': round(avg_area, 1)
        })

    return averaged

print("Detections containing center point:")
for detection in group_and_average_detections(detections):
    print(detection)
