"""
The purpose of this file is to take pictures of a calibration target.
Press "spacebar" to capture a frame and "q" to quit the program.
The calibration target should be a checkerboard pattern.
In calibrate_camera, the captured pictures will be used to get a calibration matrix.
"""
import cv2
import os
import yaml

# Read the camera number from the config file
CONFIG_PATH = "../../config.yml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
camera_num = config["camera_num"] # you may have to try 0, 1, 2, ... (depending on # of cameras on system)

# Receive the image
cap = cv2.VideoCapture(camera_num)

# Define the directory where you want to save the captured images
calibration_pictures = "calibration_pictures"
print("Saving calibration images to: ", calibration_pictures)
# Create the directory if it doesn't exist
os.makedirs(calibration_pictures, exist_ok=True)


frame_count = 0
print("Press space to capture frame. Press q to quit.")
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    # Display the frame
    cv2.imshow('Press Space to Capture, q to quit', frame)
    # Check for the spacebar key press (ASCII code 32)
    key = cv2.waitKey(1)
    if key == 32:  # 32 is the ASCII code for spacebar
        filename = os.path.join(calibration_pictures, f"frame-{frame_count:04d}.jpg")
        
        # Increment the frame count for the next captured frame
        frame_count += 1

        # Save the captured frame as a .jpg image
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")

    # Check for the 'q' key press to quit the program
    elif key == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()