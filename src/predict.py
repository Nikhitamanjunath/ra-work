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
import os

regression_file = 'regression_input/acre2-13.csv'

grouped_detections = []

import csv
import cv2

grouped_detections = []

with open('tmp/input_video.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        detection = {
            'timestamp': float(row['timestamp']),
            'truck': row['truck'].lower() == 'true',
            'width': float(row['width']),
            'height': float(row['height']),
            'area': float(row['area']),
            'image': None
        }

        image_path = row.get('image_path', '')
        if image_path:
            img = cv2.imread(image_path)
            if img is not None:
                detection['image'] = img
            else:
                print(f"Warning: could not read image at path {image_path}")

        grouped_detections.append(detection)


API_KEY="AIzaSyDHRFfN_V-YgrETeb2CoqNRitcYfrtpumg"

# Step 1: Load your CSV file
df = pd.read_csv(regression_file)

# Step 2: Prepare training data
X = df[['area', 'width', 'height']]
y = df['weight']

# Step 3: Train the regression model
regressor = LinearRegression()
regressor.fit(X, y)

def estimate_weight(detection):
    if detection['truck']:
        return 0.0
    features = np.array([[detection['area'], detection['width'], detection['height']]])
    weight_pred = regressor.predict(features)[0]
    return round(weight_pred, 2)



processedDetections = []

for detection in grouped_detections:
    weight = estimate_weight(detection)
    detection['estimated_weight_kg'] = round(weight, 1)  # Append and round weight
    processedDetections.append(detection)


# Configure the Gemini API (replace with your actual API key)
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

vehicles = []

for detection in grouped_detections:
    if detection['truck']:
        continue
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


with open('tmp/vehicles.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'name', 'weight']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for vehicle in vehicles:
        writer.writerow(vehicle)

with open('tmp/detections.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'truck', 'width', 'height', 'area', 'estimated_weight_kg']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for detection in processedDetections:
        row = {k: v for k, v in detection.items() if k != 'image'}
        writer.writerow(row)