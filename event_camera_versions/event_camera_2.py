import cv2
import numpy as np

crop_x1, crop_y1 = 200, 400
crop_x2, crop_y2 = 1200, 800

# Load the video
cap = cv2.VideoCapture("/Users/pradeeppatil/workspace/ra-work/input.mp4")
ret, prev_frame = cap.read()
if not ret:
    print("Failed to read from video.")
    exit()

prev_frame = prev_frame[crop_y1:crop_y2, crop_x1:crop_x2]
# Convert first frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Get frame dimensions
height, width = prev_frame.shape[:2]

# Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_event.avi", fourcc, 30.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge Temporal algorithm: difference between consecutive grayscale frames
    diff = cv2.absdiff(prev_gray, gray)

    # Threshold to get binary mask
    _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphological operations to reduce noise and merge blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

    # Find contours (moving object boundaries)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy original frame to draw on
    output_2 = frame.copy()
    output_frame = motion_mask.copy()
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40000:  # Filter out noise/small areas
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put area text
        text = f"Area: {int(area)}"
        cv2.putText(output_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    # Write to output video and show result
    out.write(output_frame)
    cv2.imshow("Edge Temporal Motion Detection", output_frame)
    cv2.imshow("Edge Temporal Motion Detection 2", frame)

    # Update previous frame for next iteration
    prev_gray = gray.copy()

    # Exit on 'q' key
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
