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

Helper functions for the MonoNav project.
Functionality should be concentrated here and shared between the scripts.

"""
import time
import cv2 as cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial import distance
import os
import open3d as o3d
import open3d.core as o3c
import copy
import yaml, json

# For Craziflie logging
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

# For bufferless video capture
import queue, threading

"""
VoxelBlockGrid class (adapted from Open3D) for ease of initialization and integration.
You can read more about the VoxelBlockGrid here:
https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/voxel_block_grid.html
"""
class VoxelBlockGrid:
    def __init__(self, depth_scale=1000.0, depth_max=5.0, trunc_voxel_multiplier=8.0, device=o3d.core.Device("CUDA:0")):
        # Reconstruction Information
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.trunc_voxel_multiplier = trunc_voxel_multiplier
        self.device = device
        self.camera = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault) # Kinect Intrinsics (default)
        self.depth_intrinsic = o3d.core.Tensor(self.camera.intrinsic_matrix, o3d.core.Dtype.Float64)

        # Initialize the VoxelBlockGrid
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=(1, 1, 3),
            voxel_size=3.0 / 64, # this sets the resolution of the voxel grid
            block_resolution=1,
            block_count=50000,
            device=device)

    def integration_step(self, color, depth_numpy, cam_pose):
        # Integration Step (TSDF Fusion)
        depth_numpy = depth_numpy.astype(np.uint16)  # Convert to uint16
        depth = o3d.t.geometry.Image(depth_numpy).to(self.device)
        extrinsic = o3d.core.Tensor(np.linalg.inv(cam_pose), o3d.core.Dtype.Float64)
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth, self.depth_intrinsic, extrinsic, self.depth_scale, self.depth_max, self.trunc_voxel_multiplier)
        color = o3d.t.geometry.Image(np.asarray(color)).to(self.device)
        color_intrinsic = o3d.core.Tensor(self.camera.intrinsic_matrix, o3d.core.Dtype.Float64)
        self.vbg.integrate(frustum_block_coords, depth, color, self.depth_intrinsic,
                       color_intrinsic, extrinsic, self.depth_scale, self.depth_max, self.trunc_voxel_multiplier)


"""
Bufferless VideoCapture, courtesy of Ulrich Stern (https://stackoverflow.com/a/54577746)
Otherwise, a lag builds up in the video stream.
"""
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

"""
Get the global Crazyflie (camera) pose from the logger, convert to the Open3D frame
Crazyflie frame: (X, Y, Z) is FRONT LEFT UP (FLU)
Open3D frame: (X, Y, Z) is RIGHT DOWN FRONT (RDF)
"""
def get_crazyflie_pose(scf, logstate):
    with SyncLogger(scf, logstate) as logger:
        for log_entry in logger:
            data = log_entry[1]
            _x = data['stateEstimate.x']
            _y = data['stateEstimate.y']
            _z = data['stateEstimate.z']
            _roll = data['stateEstimate.roll']
            _pitch = data['stateEstimate.pitch']
            _yaw = data['stateEstimate.yaw']
            # Convert position from CF to TSDF frame
            xyz = np.array([-_y, -_z, _x]) # Convert to TSDF frame
            # Convert rotation from CF to TSDF frame
            r = Rotation.from_euler('xyz', [_roll, -_pitch, _yaw], degrees=True)
            R = r.as_matrix()
            M_change = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
            R = M_change @ R @ M_change.T

            # Create a homogeneous matrix
            Hmtrx = np.hstack((R, xyz.reshape(3,1)))
            # return camera position
            return np.vstack((Hmtrx, np.array([0, 0, 0, 1])))

"""
Compute depth from an RGB image using ZoeDepth
Returns depth_numpy (uint16 in mm), depth_colormap (for visualization)
"""
def compute_depth(color, zoe):
    # Compute depth
    depth = zoe.infer_pil(color, output_type="tensor")  # as torch tensor
    depth_numpy = np.asarray(depth) # Convert to numpy array
    depth_numpy = 1000*depth_numpy # Convert to mm
    depth_numpy = depth_numpy.astype(np.uint16) # Convert to uint16

    # Save images and depth array
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_numpy, alpha=0.03), cv2.COLORMAP_JET)

    return depth_numpy, depth_colormap

"""
Load the poses (after navigation, for analysis) from the posedir.
Returns a list of pose arrays.
"""
def poses_from_posedir(posedir):
    poses = []
    pose_files = [name for name in os.listdir(posedir) if os.path.isfile(os.path.join(posedir, name)) and name.endswith(".txt")]
    pose_files = sorted(pose_files)

    for pose_file in pose_files:
        cam_pose = np.loadtxt(posedir +"/"+pose_file)
        poses.append(cam_pose)
    return poses

"""
Convert a list of poses (after navigation, for analysis) into a trajectory lineset.
This object is used to visualize the trajectory in Open3D.
Returns a list of of lineset objects representing the camera's pose.
"""
def get_poses_lineset(poses):
    points = []
    lines = []
    for pose in poses:
        position = pose[0:3,3] # meters
        points.append(position)
        lines.append([len(points)-1, len(points)])

    pose_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines[:-1]),
    )
    pose_lineset.paint_uniform_color([1, 0, 0]) #optional: change the color here
    return pose_lineset

"""
Load the trajectory primitives (before navigation).
Read a list of motion primitives (trajectories) from a the "trajlib_dir" (trajectory library) directory.
Returns a list of trajectory objects.
"""
def get_trajlist(trajlib_dir):
    # Get the list of files in the directory
    file_list = os.listdir(trajlib_dir)
    # Filter only .npz files
    npz_files = [file for file in file_list if file.endswith('.npz')]
    # Sort the list of .npz files - important for indexing!
    sorted_files = sorted(npz_files)
    # Iterate over the sorted list of .npz files
    traj_list = []
    for trajfile in sorted_files:
        file_path = os.path.join(trajlib_dir, trajfile)
        traj_list.append(np.load(file_path))
    
    return traj_list

"""
Convert the trajectory list into a list of trajectory linesets.
These are used for visualizing the possible trajectories at each step.
Returns a list of trajectory lineset objects.
"""
def get_traj_linesets(traj_list):
    traj_linesets = []
    amplitudes = []
    for traj in traj_list:
        # traj_dict = {key: traj[key] for key in traj.files}
        z_tsdf = traj['x_sample']
        x_tsdf = -traj['y_sample']
        points = []
        lines = []
        for i in range(len(x_tsdf)):
            points.append([x_tsdf[i], 0, z_tsdf[i]])
            lines.append([len(points)-1, len(points)])
        traj_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines[:-1]),
        )
        traj_linesets.append(traj_lineset)
        amplitudes.append(traj['amplitude'])
        # get traj info
        period = traj['period']
        forward_speed = traj['forward_speed']

    return traj_linesets, period, forward_speed, amplitudes


"""
MonoNav Planner: Return the chosen trajectory index given the current position, current reconstruction, trajectory library, and goal position.
"""
def choose_primitive(vbg, camera_position, traj_linesets, goal_position, dist_threshold, filterYvals, filterWeights, filterTSDF, weight_threshold):

    # Boolean for stopping criteria
    shouldStop = False

    # Get weights and tsdf values from the voxel block grid
    weights = vbg.attribute("weight").reshape((-1))
    tsdf = vbg.attribute("tsdf").reshape((-1))
    # Get the voxel_coords, voxel_indices
    voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices()

    # IMPORTANT
    # Use voxel_indices to rearrange weights and tsdf to match voxel_coords
    # Otherwise, the ordering of voxels from the hashmap is non-deterministic
    weights = weights[voxel_indices]
    tsdf = tsdf[voxel_indices]

    # Generate mask to filter out y values (vertical) (+y is DOWN)
    # This is useful to filter out the floor, and avoid obstacles in-plane
    if filterYvals:
        mask = voxel_coords[:, 1] < -0.3
        # Apply mask to voxel_coords and weights
        voxel_coords = voxel_coords[mask]
        weights = weights[mask]
        tsdf = tsdf[mask]

    # Generate mask to filter by weights
    # This rejects voxels below a certain weight threshold
    if filterWeights:
        mask = weights > weight_threshold
        # Apply mask to voxel_coords and weights
        voxel_coords = voxel_coords[mask,:]
        tsdf = tsdf[mask]

    # Generate mask to filter by tsdf value
    if filterTSDF:
        # Generate mask to filter by tsdf values
        mask = tsdf < 0.0
        voxel_coords = voxel_coords[mask,:]

    # transfer to cpu for cdist
    voxel_coords_numpy = voxel_coords.cpu().numpy()

    # NOW WE HAVE A FILTERED SET OF VOXELS THAT REPRESENT OBSTACLES
    # NEXT, WE DETERMINE THE BEST TRAJECTORY ACCORDING TO A COST FUNCTION

    # Initialize scoring variables to evaluate the trajectories
    max_traj_score = -np.inf # track best trajectory
    min_goal_score = np.inf # track proximity to goal
    max_traj_idx = None # track the index of the best trajectory

    # iterate over the sorted traj linesets
    for traj_idx, traj_linset in enumerate(traj_linesets):
        traj_lineset_copy = copy.deepcopy(traj_linset)
        traj_lineset_copy.transform(camera_position) # transform the lineset (copy) to the camera position
        pts = np.asarray(traj_lineset_copy.points) # meters # extract the points from the lineset
        tmp = distance.cdist(voxel_coords_numpy, pts, "sqeuclidean") # compute the distance between all voxels and all points in the trajectory
        voxel_idx, pt_idx = np.unravel_index(np.argmin(tmp), tmp.shape) # extract indices of the nearest voxel to and nearest point in the trajectory
        nearest_voxel_dist = np.sqrt(tmp[voxel_idx, pt_idx])
        if nearest_voxel_dist > dist_threshold:
            # the trajectory meets the dist_threshold criterion
            if goal_position is not None:
                # the trajectory satisfies the dist_threshold; let's compute the goal score
                tmp_to_goal = distance.cdist(goal_position, pts, "sqeuclidean")
                dst_to_goal = np.sqrt(np.min(tmp_to_goal))
                if dst_to_goal < min_goal_score:
                    # we have a trajectory that gets us closer to the goal
                    # print("traj %d gets us closer to the goal: %f"%(traj_idx, dst_to_goal))
                    max_traj_idx = traj_idx
                    min_goal_score = dst_to_goal
            else:
                # no goal position, choose the index that maximizes distance from the obstacles
                if max_traj_score < nearest_voxel_dist:
                    # we have found a trajectory that gets us closer to goal
                    max_traj_idx = traj_idx
                    max_traj_score = nearest_voxel_dist

    if max_traj_idx is None:
        # No trajectory meets the dist_threshold criterion, crazyflie should stop.
        shouldStop = True
    return shouldStop, max_traj_idx


"""
Upon Crazyflie startup, these helper functions ensure the EKF is properly initialized before takeoff.
#  Copyright (C) 2018 Bitcraze AB
Taken from several Crazyflie examples, e.g., https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/autonomy/autonomous_sequence_high_level.py
"""
def reset_estimator(scf):
    scf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.param.set_value('kalman.resetEstimation', '0')
    
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break

"""
Load config.yml file
"""
def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

"""
Read in the intrinsics.json file and return the camera matrix and distortion coefficients
"""
def get_calibration_values(camera_calibration_path):
    # Load the camera calibration file
    with open(camera_calibration_path, "r") as json_file:
        data = json.load(json_file)
    mtx = np.array(data['CameraMatrix'])
    dist = np.array(data['DistortionCoefficients'])
    return mtx, dist

"""
Transform the raw image to match the kinect image: dimensions and intrinsics.
This involves resizing the image, scaling the camera matrix, and undistorting the image.
"""
def transform_image(image, mtx, dist, kinect):
    if image.shape[0] != kinect.height or image.shape[1] != kinect.width:
        # Resize the camera matrix to match new dimensions
        scale_vec = np.array([kinect.width / image.shape[1], kinect.height / image.shape[0], 1]).reshape((3,1))
        mtx = mtx * scale_vec
        # Resize image to match the kinect dimensions & new intrinsics
        image = cv2.resize(image, (kinect.width, kinect.height))
    # Transform to the kinect camera matrix
    transformed_image = cv2.undistort(np.asarray(image), mtx, dist, None, kinect.intrinsic_matrix)
    return transformed_image

"""
Helper function to extract the image frame number from the filename string.
"""
def split_filename(filename):
    return int(filename.split("-")[-1].split(".")[0])