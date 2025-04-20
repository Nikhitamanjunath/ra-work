import cv2
import numpy as np
import time
from statistics import mean
import google.generativeai as genai
from PIL import Image
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# crop_x1, crop_y1 = 200, 400
# crop_x2, crop_y2 = 1200, 900
downsample_factor = 4  # Adjust as needed
API_KEY="AIzaSyDHRFfN_V-YgrETeb2CoqNRitcYfrtpumg"

# Load the video
video_path = "/Users/pradeeppatil/workspace/ra-work/acre.mp4"
cap = cv2.VideoCapture(video_path)

# Specify the starting timestamp in milliseconds
start_time_ms = 5000
end_time_ms = 95000
# start_time_ms = 9000
# end_time_ms = 15000
fps = cap.get(cv2.CAP_PROP_FPS)
end_frame = int(end_time_ms / 1000 * fps)

# Set the video capture to the desired starting point
cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)
ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (prev_frame.shape[1] // downsample_factor, prev_frame.shape[0] // downsample_factor))

if not ret:
    print("Failed to read from video.")
    exit()

# prev_frame = prev_frame[crop_y1:crop_y2, crop_x1:crop_x2]
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
height, width = prev_gray.shape[:2]

# Output video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_event.avi", fourcc, 30.0, (width, height))

# List to store detections
detections = []

x_offset = 0
y_offset = 115
center_x, center_y = ((width // 2) + x_offset), ((height // 2) + y_offset)

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame.shape[1] // downsample_factor, frame.shape[0] // downsample_factor))
    # frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_2 = frame.copy()
    output_frame = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    cushion = 50
    cv2.rectangle(
        output_frame,
        (center_x - cushion, center_y - cushion),
        (center_x + cushion, center_y + cushion),
        (255, 0, 0), 2
    )
    cv2.rectangle(
        frame,
        (center_x - cushion, center_y - cushion),
        (center_x + cushion, center_y + cushion),
        (255, 0, 0), 2
    )

    # Optional: Draw the actual center point of the screen
    cv2.circle(output_frame, (center_x, center_y), 5, (255, 255, 0), -1)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 18000:
            continue
        if area > 50000:
            timestamp = round(time.time() - start_time, 2)
            detection = {
                "timestamp": timestamp,
                "truck": True,
                "width": 0,
                "height": 0,
                "area": area,
                "image": None
            }

            detections.append(detection)
            print(detection)
            continue
            

        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add area text
        cv2.putText(output_frame, f"Area: {int(area)}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Calculate center of the bounding box
        box_cx = x + w // 2
        box_cy = y + h // 2

        # Optional: Draw center point of the box
        cv2.circle(output_frame, (box_cx, box_cy), 5, (0, 0, 255), -1)

        # Check if box center is close to screen center (within a few pixels)
        if abs(box_cx - center_x) <= cushion and abs(box_cy - center_y) <= cushion:
            timestamp = round(time.time() - start_time, 2)

            # Crop the region around the bounding box from the original frame
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            bbox_image = frame[y1:y2, x1:x2].copy()  # Make a copy of the cropped region

            detection = {
                "timestamp": timestamp,
                "truck": False,
                "width": w,
                "height": h,
                "area": area,
                "image": bbox_image  # Add the image here
            }
            detections.append(detection)
            print(f"[{timestamp}s] Detection with area {area} and image shape {bbox_image.shape}")

    out.write(output_frame)
    cv2.imshow("Edge Temporal Motion Detection", output_frame)
    cv2.imshow("Original Frame", frame)

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if current_frame > end_frame:
        print("Reached the specified end time.")
        break

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
            'truck': group[0]['truck'],
            'width': round(avg_w, 1),
            'height': round(avg_h, 1),
            'area': round(avg_area, 1),
            'image': group[len(group) // 2]['image']
        })

    return averaged

# Step 1: Load your CSV file
df = pd.read_csv('vehicles3.csv')

# Step 2: Prepare training data
X = df[['area', 'width', 'height']]
y = df['weight']

# Step 3: Train the regression model
regressor = LinearRegression()
regressor.fit(X, y)

# Optional: Save the model to disk if you want to reuse it later
# joblib.dump(regressor, 'weight_regressor.pkl')

# Step 4: Define a new estimate_weight function using regression
def estimate_weight(detection):
    if detection['truck']:
        return 0.0
    features = np.array([[detection['area'], detection['width'], detection['height']]])
    weight_pred = regressor.predict(features)[0]
    return round(weight_pred, 2)

# Example usage:
# Replace this with your actual detection dictionary
sample_detection = {
    'timestamp': 200.00,
    'truck': False,
    'width': 350,
    'height': 110,
    'area': 23000
}

processedDetections = []

for detection in group_and_average_detections(detections):
    weight = estimate_weight(detection)
    detection['estimated_weight_kg'] = round(weight, 1)  # Append and round weight
    processedDetections.append(detection)


processedDetections2 = []
truck_count = 0
for d in processedDetections:
    if not d['truck']:
        processedDetections2.append(d)
    else:
        truck_count += 1

print("Total Trucks: ", truck_count)


# for detection in processedDetections:
#     if not detection['truck'] and detection['image'].any():
#         cv2.imshow(f'Detection ', detection['image'])
#         key = cv2.waitKey(0)  # Wait for key press to show each image one at a time
#         cv2.destroyWindow(f'Detection ')


# Configure the Gemini API (replace with your actual API key)
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

vehicles = []

for detection in processedDetections2:
    if detection['truck']:
        vehicles.append("truck")
    if not detection['truck']:
        image = detection['image']
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            contents = [
                pil_image,
                "Identify the make and model of the vehicle in this image and provide an estimated weight range in kilograms. Respond in the format ' Name: (Vehicle Name) Weight: (x)kg '. Strictly do not deviate from output format and if there is a weight range pick median of that weight range for x and only respond in that format"
            ]
            response = model.generate_content(contents)
            response_text = response.text.strip()
            print(f"Gemini API Response: {response_text}")

            vehicle_name = "Unknown"
            weight_value = "Unknown"

            if response_text.startswith("Name:") and "Weight:" in response_text:
                try:
                    name_part, weight_part = response_text.split("Weight:")
                    vehicle_name = name_part.split("Name:")[-1].strip()
                    weight_str = weight_part.replace("kg", "").strip()
                    if "-" in weight_str:
                        low, high = map(float, weight_str.split("-"))
                        weight_value = (low + high) / 2
                    else:
                        weight_value = float(weight_str)
                except ValueError:
                    print(f"Could not parse response: {response_text}")
                    vehicle_name = "Error"
                    weight_value = 0
                except Exception as e:
                    print(f"Error during parsing: {e}")
                    vehicle_name = "Error"
                    weight_value = 0
            else:
                print(f"Unexpected response format: {response_text}")
                vehicle_name = "Format Error"
                weight_value = 0

            vehicles.append({
                'timestamp': detection['timestamp'],
                'name': vehicle_name,
                'weight': f"{weight_value}" if isinstance(weight_value, (int, float)) else weight_value
            })

        except Exception as e:
            print(f"Error processing image with Gemini API: {e}")
            vehicles.append({
                'name': "Error",
                'weight': 0
            })

print("\nVehicle Information:")

for vehicle_info in vehicles:
    print(f"Name: {vehicle_info['name']}")
    print(f"Weight: {vehicle_info['weight']}")

with open('vehicles.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'name', 'weight']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for vehicle in vehicles:
        writer.writerow(vehicle)

with open('detections.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'truck', 'width', 'height', 'area', 'estimated_weight_kg']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for detection in processedDetections2:
        row = {k: v for k, v in detection.items() if k != 'image'}
        writer.writerow(row)
# for detection, vehicle in zip(processedDetections,vehicles):
#     error = float(detection['estimated_weight_kg']) - float(vehicle['weight'])
#     print(detection['timestamp'], " ---- ", vehicle['name'], " ---- ", error)

# errors = np.abs(np.array([float(d['estimated_weight_kg']) for d in processedDetections]) - np.array([float(v['weight']) for v in vehicles]))
# mae = np.mean(errors)
# print(f"Mean Absolute Error (MAE): {mae}")

# squared_errors = (np.array([float(d['estimated_weight_kg']) for d in processedDetections]) - np.array([float(v['weight']) for v in vehicles])) ** 2
# mse = np.mean(squared_errors)
# print(f"Mean Squared Error (MSE): {mse}")

# rmse = np.sqrt(mse)
# print(f"Root Mean Squared Error (RMSE): {rmse}")

# medae = np.median(errors)
# print(f"Median Absolute Error (MedAE): {medae}")

# actual_weights = np.array([float(v['weight']) for v in vehicles])
# predicted_weights = np.array([float(d['estimated_weight_kg']) for d in processedDetections])
# mean_actual = np.mean(actual_weights)
# ss_total = np.sum((actual_weights - mean_actual) ** 2)
# ss_residual = np.sum((actual_weights - predicted_weights) ** 2)
# r_squared = 1 - (ss_residual / ss_total)
# print(f"R-squared (Coefficient of Determination): {r_squared}")

# actual_weights = np.array([float(v['weight']) for v in vehicles])
# predicted_weights = np.array([float(d['estimated_weight_kg']) for d in processedDetections])
# mape = np.mean(np.abs((actual_weights - predicted_weights) / actual_weights)) * 100
# print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# # The following lines for displaying images are kept as they were,
# # but the vehicle information is now being extracted using the Gemini API.
# # for detection in processedDetections:
# #     if not detection['truck'] and detection['image'].any():
# #         cv2.imshow(f'Detection ', detection['image'])
# #         key = cv2.waitKey(0)  # Wait for key press to show each image one at a time
# #         cv2.destroyWindow(f'Detection ')