"""
This script tests the camera feed.
"""

import cv2
import yaml

CONFIG_PATH = "../config.yml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# If you have multiple cameras hooked up to your desktop,
# The camera number may change. If so, try
# a different small, positive integer, e.g. -1, 0, 1, 2, 3, ...
# If you're still having issues, ensure you have ffmpeg installed.

camera = config["camera_num"]
cap = cv2.VideoCapture(camera)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        print("No frame captured.")
        break
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Hit q to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()