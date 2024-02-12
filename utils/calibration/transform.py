"""
This script is used to transform the images from the robot's camera to the kinect's camera.
It uses the camera intrinsics from calibrate.py, and OpenCV's undistort function.
The transform_image function also resizes the image (and intrinsic matrix) appropriately
to match the kinect's dimensions and intrinsics. This is crucial for the performance of
the depth estimation model.
"""

import os
import open3d as o3d
import numpy as np
import cv2

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, get_calibration_values, transform_image

# LOAD CONFIG
CONFIG_PATH = "../../config.yml"
config = load_config(CONFIG_PATH)

# Load the calibration images from "calibration_pictures"
calibration_dir = "calibration_pictures"
print("Loading images from: ", calibration_dir)
calibration_files = [file for file in os.listdir(calibration_dir) if file.endswith('.jpg')]
calibration_files = sorted(calibration_files)

# Load the calibration values
camera_calibration_path = "intrinsics.json"#os.path.join('../../',config["camera_calibration_path"])
intrinsic_filename = os.path.basename(camera_calibration_path)
print("Loading intrinsics from: ", intrinsic_filename)
mtx, dist = get_calibration_values(camera_calibration_path) # for the robot's camera
# Kinect intrinsic matrix
kinect = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)


# Set directories for images
transform_dir = "transform_output"
os.mkdir(transform_dir) if not os.path.exists(transform_dir) else None
print("Saving transformed images to: ", transform_dir)


for filename in calibration_files:
    # Read in the image
    img = cv2.imread(os.path.join(calibration_dir, filename))
    # transform
    transformed_image = transform_image(np.asarray(img), mtx, dist, kinect)
    # write image
    cv2.imwrite(os.path.join(transform_dir, filename), transformed_image)