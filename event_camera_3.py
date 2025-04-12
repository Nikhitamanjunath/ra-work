import cv2
import numpy as np
import time
from statistics import mean

crop_x1, crop_y1 = 200, 400
crop_x2, crop_y2 = 1200, 800

# Load the video
cap = cv2.VideoCapture("/Users/pradeeppatil/workspace/ra-work/input.mp4")
ret, prev_frame = cap.read()
if not ret:
    print("Failed to read from video.")
    exit()

prev_frame = prev_frame[crop_y1:crop_y2, crop_x1:crop_x2]
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
height, width = prev_frame.shape[:2]

# Output video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_event.avi", fourcc, 30.0, (width, height))

# List to store detections
detections = []

# Center point of cropped frame
center_x, center_y = width // 2, height // 2

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_2 = frame.copy()
    output_frame = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add area text
        cv2.putText(output_frame, f"Area: {int(area)}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Check if center point is inside bounding box
        if x <= center_x <= x + w and y <= center_y <= y + h:
            timestamp = round(time.time() - start_time, 2)
            detection = {
                "timestamp": timestamp,
                "width": w,
                "height": h,
                "area": area
            }
            detections.append(detection)
            print(detection)


    out.write(output_frame)
    cv2.imshow("Edge Temporal Motion Detection", output_frame)
    cv2.imshow("Original Frame", frame)

    prev_gray = gray.copy()
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

    for i in range(1, len(d)):
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

# Print collected detections
print("Detections containing center point:")
for detection in group_and_average_detections(detections):
    print(detection)