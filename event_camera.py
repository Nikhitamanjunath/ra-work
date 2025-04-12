import cv2
import numpy as np

cap = cv2.VideoCapture("/Users/pradeeppatil/workspace/ra-work/input.mp4")
# cap = cv2.VideoCapture("C:/Users/nikhi/Documents/workspace/python/niks/input.mp4")
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_event.avi", fourcc, 30.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute pixel-wise difference (simulate event trigger)
    diff = cv2.absdiff(prev_gray, gray)
    
    # Threshold to highlight motion
    _, event_map = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Convert to 3-channel for visualization
    event_map_colored = cv2.cvtColor(event_map, cv2.COLOR_GRAY2BGR)
    
    out.write(event_map_colored)
    cv2.imshow("Event Camera Simulation", event_map_colored)

    prev_gray = gray.copy()
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
