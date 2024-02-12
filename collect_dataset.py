"""
  __  __                   _   _             
 |  \/  | ___  _ __   ___ | \ | | __ ___   __
 | |\/| |/ _ \| '_ \ / _ \|  \| |/ _` \ \ / /
 | |  | | (_) | | | | (_) | |\  | (_| |\ V / 
 |_|  |_|\___/|_| |_|\___/|_| \_|\__,_| \_/  
Copyright (c) 2023 Nate Simon
License: MIT
Authors: Nate Simon and Anirudha Majumdar, Princeton University
Project Page: https://natesimon.github.io/mononav

The purpose of this script is to collect a custom dataset for the MonoNav demo.
This script saves images and poses from a Crazyflie to file.
Those images and poses can then be used in the demo (depth estimation, fusion, and simulated planning).

"""
import cv2
import numpy as np
import time
import os
import time

# For Craziflie logging
import logging
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

# sys.path.append('/home/nsimon/Documents/MonoNav/') 
from utils.utils import reset_estimator, VideoCapture, get_crazyflie_pose, load_config
'''
This file collects the following synchronized images:
- Crazyflie (front-facing camera)
- Crazyflie pose
- Timestamps
'''
# LOAD CONFIG
CONFIG_PATH = "config.yml"
config = load_config("config.yml")

URI = uri_helper.uri_from_env(default=config["radio_uri"])
logging.basicConfig(level=logging.ERROR)
height = config["height"]
FLY_CRAZYFLIE = config["FLY_CRAZYFLIE"]
camera_num = config["camera_num"]
    
# Make directories for data
save_dir = 'data/trial-' + time.strftime("%Y-%m-%d-%H-%M-%S")
os.mkdir(save_dir) if not os.path.exists(save_dir) else None

crazyflie_img_dir = os.path.join(save_dir, "crazyflie-rgb-images")
crazyflie_pose_dir = os.path.join(save_dir, "crazyflie-poses")

os.mkdir(crazyflie_img_dir) if not os.path.exists(crazyflie_img_dir) else None
os.mkdir(crazyflie_pose_dir) if not os.path.exists(crazyflie_pose_dir) else None

print("Saving files to: " + save_dir)

# Drone object
# Initialize the low-level drivers
cflib.crtp.init_drivers()

# Set up log conf
logstate = LogConfig(name='state', period_in_ms=10)
logstate.add_variable('stateEstimate.x', 'float')
logstate.add_variable('stateEstimate.y', 'float')
logstate.add_variable('stateEstimate.z', 'float')
logstate.add_variable('stateEstimate.roll', 'float')
logstate.add_variable('stateEstimate.pitch', 'float')
logstate.add_variable('stateEstimate.yaw', 'float')

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    cf = scf.cf

    reset_estimator(cf)

    # Camera Object
    cap = VideoCapture(camera_num)

    # Initialize counter for images
    frame_number = 0       

    while True:

        start_time = time.time()
        
        crazyflie_rgb = cap.read()

        # Save images and depth array
        cv2.imwrite(crazyflie_img_dir + "/crazyflie_frame-%06d.rgb.jpg"%(frame_number), crazyflie_rgb)
        camera_position = get_crazyflie_pose(scf, logstate)
        np.savetxt(crazyflie_pose_dir + "/crazyflie_frame-%06d.pose.txt"%(frame_number), camera_position)

        # Update counter
        frame_number += 1 

        # Show images
        cv2.imshow('crazyflie', crazyflie_rgb)
        if chr(cv2.waitKey(1)&255) == 'q':
            break
        time.sleep(0.1)
cap.cap.release()
cv2.destroyAllWindows()