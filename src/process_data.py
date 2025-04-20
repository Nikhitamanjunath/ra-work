import csv
import numpy as np

# Load vehicles (actual weights)
vehicles = []
folder = "reg"
with open(folder+'/vehicles.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vehicles.append({
            'timestamp': row['timestamp'],
            'weight': float(row['weight'])
        })

# Load detections (predicted weights)
processedDetections = []
with open(folder+'/detections.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # This assumes `estimated_weight_kg` was saved in averaged.csv
        if 'estimated_weight_kg' in row:
            estimated_weight = float(row['estimated_weight_kg'])
        else:
            # Estimate from area as fallback
            estimated_weight = float(row['area']) * 0.073

        processedDetections.append({
            'timestamp': row['timestamp'],
            'estimated_weight_kg': estimated_weight
        })

# Make sure both lists are aligned by timestamp
# You can sort or match closest values here if needed
vehicles.sort(key=lambda x: x['timestamp'])
processedDetections.sort(key=lambda x: x['timestamp'])

# Ensure they are equal in length (trim the longer one)
min_len = min(len(vehicles), len(processedDetections))
vehicles = vehicles[:min_len]
processedDetections = processedDetections[:min_len]

# Compute error metrics
actual_weights = np.array([v['weight'] for v in vehicles])
predicted_weights = np.array([d['estimated_weight_kg'] for d in processedDetections])

errors = np.abs(predicted_weights - actual_weights)
mae = np.mean(errors)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

mse = np.mean((predicted_weights - actual_weights) ** 2)
print(f"Mean Squared Error (MSE): {mse:.2f}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

medae = np.median(errors)
print(f"Median Absolute Error (MedAE): {medae:.2f}")

# R-squared
mean_actual = np.mean(actual_weights)
ss_total = np.sum((actual_weights - mean_actual) ** 2)
ss_residual = np.sum((actual_weights - predicted_weights) ** 2)
r_squared = 1 - (ss_residual / ss_total)
print(f"R-squared (Coefficient of Determination): {r_squared:.4f}")

# MAPE
mape = np.mean(np.abs((actual_weights - predicted_weights) / actual_weights)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
