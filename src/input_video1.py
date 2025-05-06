
import cv2
import numpy as np
import time
from statistics import mean
import os
import csv

# Downsample factor
downsample_factor = 4

# Load the video
video_path = "C:/Users/nikhi/Documents/workspace/ra-work/acre.mp4"
cap = cv2.VideoCapture(video_path)

# Specify the starting timestamp in milliseconds
start_time_ms = 8000
end_time_ms = 15000000

fps = cap.get(cv2.CAP_PROP_FPS)
end_frame = int(end_time_ms / 1000 * fps)

# Set the video capture to the desired starting point
cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

ret, prev_frame = cap.read()

if not ret or prev_frame is None:
    print("Failed to read from video.")
    exit()

prev_frame = cv2.resize(prev_frame, (prev_frame.shape[1] // downsample_factor, prev_frame.shape[0] // downsample_factor))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
height, width = prev_gray.shape[:2]

detections = []
x_offset = 0
y_offset = 115
center_x, center_y = ((width // 2) + x_offset), ((height // 2) + y_offset)

start_time = time.time()
object_id = 0
prev_centers = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame.shape[1] // downsample_factor, frame.shape[0] // downsample_factor))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_frame = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    cushion = 50
    cv2.rectangle(output_frame, (center_x - cushion, center_y - cushion),
                  (center_x + cushion, center_y + cushion), (255, 0, 0), 2)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 18000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2

        found = False
        for oid, (pcx, pcy) in prev_centers.items():
            if abs(pcx - cx) < 50 and abs(pcy - cy) < 50:
                object_id = oid
                found = True
                break

        if not found:
            object_id += 1

        if object_id in prev_centers:
            pcx, pcy = prev_centers[object_id]
            if cx > pcx:
                cv2.arrowedLine(frame, (pcx, pcy), (cx, cy), (0, 0, 255), 2, tipLength=0.4)

        prev_centers[object_id] = (cx, cy)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {object_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if abs(cx - center_x) <= cushion and abs(cy - center_y) <= cushion:
            timestamp = round(time.time() - start_time, 2)
            bbox_image = frame[y:y + h, x:x + w].copy()
            detection = {
                "timestamp": timestamp,
                "truck": False,
                "width": w,
                "height": h,
                "area": area,
                "image": bbox_image
            }
            detections.append(detection)
            print(f"[{timestamp}s] Detection with area {area} and image shape {bbox_image.shape}")

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if current_frame > end_frame:
        print("Reached the specified end time.")
        break

    prev_gray = gray.copy()
    cv2.imshow("Motion with Arrows", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# --- Save outputs ---
def group_and_average_detections(d, time_window=5.0):
    if not d:
        return []

    d = sorted(d, key=lambda x: x['timestamp'])
    grouped = []
    current_group = [d[0]]

    for i in range(1, len(d)):
        if d[i]['timestamp'] - current_group[0]['timestamp'] <= time_window:
            current_group.append(d[i])
        else:
            grouped.append(current_group)
            current_group = [d[i]]
    if current_group:
        grouped.append(current_group)

    averaged = []
    for group in grouped:
        avg_w = mean([item['width'] for item in group])
        avg_h = mean([item['height'] for item in group])
        avg_area = mean([item['area'] for item in group])
        mid_timestamp = group[len(group) // 2]['timestamp']
        averaged.append({
            'timestamp': round(mid_timestamp, 2),
            'truck': group[0]['truck'],
            'width': round(avg_w, 1),
            'height': round(avg_h, 1),
            'area': round(avg_area, 1),
            'image': group[len(group) // 2]['image']
        })
    return averaged

grouped_detections = group_and_average_detections(detections)

output_dir = 'detection_images'
csv_output_dir = 'tmp'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

with open('tmp/input_video.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'truck', 'width', 'height', 'area', 'image_path']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i, detection in enumerate(grouped_detections):
        image_path = ''
        if 'image' in detection and detection['image'] is not None:
            image_path = os.path.join(output_dir, f"detection_{i}.png")
            cv2.imwrite(image_path, detection['image'])
        row = {
            'timestamp': detection['timestamp'],
            'truck': detection['truck'],
            'width': detection['width'],
            'height': detection['height'],
            'area': detection['area'],
            'image_path': image_path
        }
        writer.writerow(row)

truck_count = sum(1 for d in grouped_detections if d['truck'])
with open('tmp/trucks.txt', 'w') as f:
    f.write(f"Total Trucks: {truck_count}\n")
print("Total Trucks:", truck_count)
