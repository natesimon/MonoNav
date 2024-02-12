# MonoNav: MAV Navigation via Monocular<br>Depth Estimation and Reconstruction

**Authors**: [Nathaniel Simon](https://natesimon.github.io/) and [Anirudha Majumdar](https://irom-lab.princeton.edu/majumdar/)

_[Intelligent Robot Motion Lab](https://irom-lab.princeton.edu/), Princeton University_

[Project Page](https://natesimon.github.io/mononav) | [Paper (arXiv)](https://arxiv.org/pdf/2311.14100.pdf) | [Video](https://www.youtube.com/watch?v=msWLSfOmTpI) |

---

MonoNav is a monocular navigation stack that uses RGB images and camera poses to generate a 3D reconstruction, enabling the use of conventional planning techniques. MonoNav leverages pre-trained depth-estimation ([ZoeDepth](https://github.com/isl-org/ZoeDepth)) and off-the-shelf fusion ([Open3D](https://github.com/isl-org/Open3D)) to generate a real-time 3D reconstruction of the environment. At each planning step, MonoNav selects from a library of motion primitives to navigate collision-free towards the goal. While the robot executes each motion primitive, new images and poses are integrated into the reconstruction. In our [paper](https://arxiv.org/pdf/2311.14100.pdf), we demonstrate MonoNav on a 37 gram micro aerial vehicle (MAV) navigating hallways at 0.5 m/s (see [project page](https://natesimon.github.io/mononav) and [video](https://www.youtube.com/watch?v=msWLSfOmTpI)).

## Overview
<img src="utils/reconstruction.gif" height="250px" align="right"/>


This repository contains code to run the following:  
**MonoNav pipeline** ([`mononav_cf.py`](mononav_cf.py)): Monocular hallway navigation using a Crazyflie + FPV camera, as seen in our paper. We encourage you to adapt this script to other vehicles / scenes!

We also offer scripts that break MonoNav into sub-parts, which can be run independently & offline:
1. **Data collection pipeline** ([`collect_dataset.py`](collect_dataset.py)): Collect images and poses from your own camera / robot.
2. **Depth estimation pipeline** ([`estimate_depth.py`](estimate_depth.py)): Estimate depths from RGB images using ZoeDepth.
3. **Fusion pipeline** ([`fuse_depth.py`](fuse_depth.py)): Fuse the estimated depth images and camera poses into a 3D reconstruction.
4. **Simulate MonoNav** ([`simulate.py`](simulate.py)): Step through the 3D reconstruction and visualize the motion primitives chosen by the MonoNav planner. This is a useful way to replay and debug MonoNav trials.

These scripts (run in sequence) form a demo, which we highly recommend before adapting MonoNav for your system. We include a sample dataset ([`data/demo_hallway`](data/demo_hallway)), so no robot is needed to run the demo!

In addition, we include the following resources in the `/utils/` directory:
- [`utils/test_camera.py`](utils/test_camera.py) to test your camera,
- [`utils/generate_primitives.py`](utils/generate_primitives.py) to generate and visualize new motion primitives,
- [`utils/calibration/take_pictures.py`](utils/calibration/take_pictures.py) to take pictures of a calibration target,
- [`utils/calibration/calibrate.py`](utils/calibration/calibrate.py) to calibrate the camera with OpenCV and save the camera intrinsics to file,
- [`utils/calibration/transform.py`](utils/calibration/transform.py) to test the undistortion and transformation pipeline.

We hope you enjoy MonoNav!

## Installation and Configuration

Clone the repository and its submodules (ZoeDepth):

```
git clone --recurse-submodules https://github.com/natesimon/MonoNav.git
```

Install the dependencies from `environment.yml` using [mamba](https://github.com/mamba-org/mamba) (fastest):
```bash
mamba env create -n mononav --file environment.yml
mamba activate mononav
```
or conda : 

```bash
conda env create -n mononav --file environment.yml
conda activate mononav
```
**Note:** If the installation gets stuck at `Solving environment: ...`, we recommend updating your system, re-installing conda / miniconda, or using mamba.

**Tested on:** (release / driver / GPU)  
- Ubuntu 22.04 / NVIDIA 535 / RTX 4090
- Ubuntu 22.04 / NVIDIA 535 / Titan RTX
- Ubuntu 20.04 / NVIDIA 530 / Titan Xp
- Ubuntu 18.04 / NVIDIA 470 / Titan Xp

If you do not have access to GPU, set `device = "CPU:0"` in `config.yml`. This will reduce the speed of both depth estimation and fusion, and may not be fast enough for real-time operations.

## Demo: Out of the Box!

We include a demo dataset ([`data/demo_hallway`](data/demo_hallway)) to try MonoNav out of the box - no robot needed! From a series of ([occasionally noisy](data/demo_hallway/crazyflie-rgb-images/crazyflie_frame-000030.rgb.jpg)) images and poses, we will transform the images, estimate depth, and fuse them into a [3D reconstuction](utils/reconstruction.gif).
1. The dataset includes just RGB images and camera poses from a Crazyflie (see [Hardware](#mononav-hardware)):
    ```
    ├── <demo_hallway>
    │   ├── <crazyflie_poses> # camera poses
    │   ├── <crazyflie_rgb_images> # raw camera images
    ```
1. Set the dataset path in `config.yml`.  By default, `data_dir: 'data/demo_hallway`, but make sure to change this if you want to process your own dataset.
    ```
    data_dir: 'data/demo_hallway' # change to whichever directory you want to process
    ```
1. To demonstrate ZoeDepth, run `python estimate_depth.py`. This reads in the crazyflie images and transforms them to match the camera intrinsics used in the ZoeDepth training dataset. This is crucial for depth estimation accuracy (see [Calibration](#camera-calibration) for more details). The transformed images are saved in `<kinect_rgb_images>` and used to estimate depth. The estimated depths are saved as numpy arrays and colormaps (for visualization) in `<kinect_depth_images>`. After running, take a look at the resulting images and note the loss of peripheral information as the raw images are undistorted.
1. To demonstrate fusion, run: `python fuse_depth.py`. This script reads in the (transformed) images, poses, and depths, and integrates them using Open3D's TSDF Fusion. After completion, a reconstruction should be displayed with coordinate frames to mark the camera poses throughout the run. The reconstruction is saved to file as a VoxelBlockGrid (`vbg.npz`) and pointcloud (`pointcloud.ply` - which can be opened using MeshLab).
1. Next, run `python simulate.py`. This loads the reconstruction (`vbg.npz`) and executes the MonoNav planner. The planner is executed at each of the camera poses, and does the following:
    1. visualizes (in black) the available motion primitives in the trajectory library (`utils/trajlib`),
    1. chooses a motion primitive according to the planner: `choose_primitive()` in `utils/utils.py` selects the primitive that makes the most progress towards `goal_position` while remaining `min_dist2obs` from all obstacles in the reconstruction,
    1. paints the chosen primitive green.
`simulate.py` is useful for debugging and de-briefing, and also to anticipate how changes in the trajectory library or planner affect performance. For example, by changing `min_dist2obs` in`config.yml`, it is possible to see how increasing/decreasing the distance threshold to obstacles affects planner performance.

1. Finally, try changing the motion primitives to see how they affect planner performance! To modify and generate the trajectory library, open `utils/generate_primitives.py`. Try changing `num_trajectories` from `7` to `11`, and run `generate_primitives.py.` This will display the new motion primitives and update the trajectory library. Note that each motion primitive is defined by a set of gentle turns left, right, or straight. An "extension" segment is added to the primitive (but not flown) to encourage foresight in the planner. See our paper for more details. Feel free to re-run `simulate.py` to try out the new primitives!

The tutorial should result in the additional files added to `data/demo_hallway`:
```
├── <demo_hallway>
│   ├── <kinect_rgb_images> # images transformed to match kinect intrinsics
│   ├── <kinect_depth_images> # estimated depth (.npy for fusion and .jpg for visualization)
│   ├── vbg.npz / pointcloud.ply # reconstructions generated by fuse_depth.py
```
If you are unable to execute the full tutorial but want to reference the output, you can download it from [Google Drive](https://drive.google.com/file/d/1r9KgkOoOsSP_7FARzuB4NcOaF94TS1HF/view?usp=sharing). If you have made it through the tutorials, you can try MonoNav on your own dataset!

## Collect your own Dataset

To run MonoNav on your own dataset, there are two crucial steps:
1. Collect your own dataset (RGB images and poses). We provide [`collect_dataset.py`](collect_dataset.py) which works for the Crazyflie, but you may have to modify it for your system. Ensure that you are transforming the pose (rotation + translation) into the Open3D frame correctly, and saving it in homogeneous form. See `get_crazyflie_pose` in `utils/utils.py` for reference. Make sure to update `data_dir` in `config.yml` to point to your collected dataset.
2. Provide the camera intrinsics. This is crucial for the image transformation step and can affect depth estimation quality dramatically. We recommend that you follow our provided calibration sequence to automatically generate a `.json` of camera intrinsics and distortion coefficients. `config.yml` should then be updated to point to the intrinsics json file path: `camera_calibration_path: 'utils/calibration/intrinsics.json'`.

With those steps complete, you can run `estimate_depth.py`, `fuse_depth.py`, and `simulate.py` to reconstruct and try MonoNav on your own dataset!

## Camera Calibration
A key aspect to MonoNav is using a pre-trained depth estimation network on a different camera than the one used during training. The micro FPV camera ([Wolfwhoop WT05](https://a.co/d/bhTwelB)) that we use has significant barrel distortion (fish-eye), and thus the images must be first undistorted to better match the camera intrinsics used to collect the training data.  To maintain the metric depth estimation accuracy of the model, we must transform the input image to match the intrinsics of the training dataset. ZoeDepth is trained on [NYU-Depth-v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), which used the Microsoft [Kinect](https://en.wikipedia.org/wiki/Kinect).

The `transform_image()` function in `utils/utils.py` performs the transformation: resizing the image and undistorting it to match the Kinect's intrinsics.

In `utils/calibration`, we provide scripts to generate `intrinsics.json` for your own camera. Steps to calibrate:

1. Make a chessboard calibration target. 
1. `take_pictures.py`: Take many pictures (recommended: 80+) of the chessboard by pressing the spacebar. Saves them to `utils/calibration/calibration_pictures/`.
1. `calibrate.py`: Based on the [OpenCV sample](https://github.com/opencv/opencv/blob/4.x/samples/python/calibrate.py).You need to provide several arguments, including the structure and dimensions of your chessboard target. Example:
    ```
    MonoNav/utils/calibration$ python calibrate.py -w 6 -h 8 -t chessboard --square_size=35 ./calibration_pictures/frame*.jpg

    ```
    The intrinsics are printed and saved to `utils/calibration/intrinsics.json`.
1. `transform.py`: This script loads the intrinsics from `intrinsics.json` and transforms your `calibration_pictures` to the Kinect's dimensions (640x480) and intrinsics. This operation may involve resizing your image. The transformed images are saved in `utils/calibration/transform_output` and should be inspected.
1. Finally, we recommend re-running calibration on `transform_output` to ensure that the intrinsics match the Kinect.
    ```
    MonoNav/utils/calibration$ python calibrate.py -w 6 -h 8 -t chessboard --square_size=35 ./transform_output/frame*.jpg

    ```
    `transform.py` will save the intrinsics of the transformed images to `check_intrinsics.json`, which should roughly match those of the Kinect:
    ```
    [[525.    0.  319.5]
    [  0.  525.  239.5]
    [  0.    0.    1. ]]
    ```
    
## MonoNav: Hardware
To run MonoNav as shown in our paper, you need a monocular robot with pose (position & orientation) estimation. We used the [Crazyflie 2.1](https://www.bitcraze.io/products/crazyflie-2-1/) micro aerial vehicle modified with an FPV camera. Our hardware setup follows closely the one used in Princeton's [Introduction to Robotics](https://irom-lab.princeton.edu/intro-to-robotics/) course. If you are using the Crazyflie, we recommend that you follow the [Bitcraze tutorials](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/) to ensure that the vehicle flies and commmunicates properly.

**List of parts:**
- [Crazyflie 2.1](https://www.bitcraze.io/products/crazyflie-2-1/) micro aerial vehicle,
- [Flowdeck v2](https://www.bitcraze.io/products/flow-deck-v2/) for position and velocity estimation (pose),
- 5.8 GHz micro FPV camera (e.g., [Wolfwhoop WT05](https://a.co/d/bhTwelB)),
- Custom PCB with [long pins](https://store.bitcraze.io/collections/spare-parts-crazyflie-2-0/products/male-long-deck-connector), to attach the camera to the Crazyflie,
- [Crazyradio PA](https://store.bitcraze.io/collections/kits/products/crazyradio-pa) for communication with the vehicle,
- 5.8 GHz video receiver (e.g., [Skydroid](https://a.co/d/9lO7Md8)).

## Running `mononav_cf.py`

The `mononav_cf.py` ("cf" for "crazyflie") script performs the image transformation, depth estimation, fusion, and planning process simultaneously for goal-directed obstacle avoidance. If `FLY_CRAZYFLIE: True`, the Crazyflie will takeoff, if `False`, the pipeline will execute without the Crazyflie starting its motors (useful for testing).

After takeoff, the Crazyflie can be controlled manually by the following key commands:
```
w: choose MIDDLE primitive (typically FORWARD)
a: choose FIRST primitive (typically LEFT)
d: choose LAST primitive (typically RIGHT)
c: end control (stop and land)
q: end control immediately (EMERGENCY stop and land)
g: start MonoNav
```

Manual control is an excellent way to check that the pipeline is working, as it should produce a sensible reconstruction after landing. As mentioned in the paper, it is HIGHLY RECOMMENDED to manually fly forward 3x (press `w, w, w`) before starting MonoNav (press `g`). This is due to the narrow field of view of the transformed images, which discards peripheral information; to make an informed decision, the planner needs information collected 3 primitives ago.

After MonoNav is started, the Crazyflie will choose and execute primitives according to the planner. If collision seems imminent, you can manually choose a primitive (by `w, a, d`) or stop the planner (`c` or `q`). During the run, a `crazyflie_trajectories.csv` log is produced, which includes the `frame_number` and `time_elapsed` during replanning, as well as the `chosen_traj_idx`. 

## Future Work

MonoNav is a "work in progress" - there are many exciting directions for future work! If you find any bugs or implement any exciting features, please submit a pull request as we'd love to continue improving the system. Once you get MonoNav running on your robot, send us a video! We'd love to see it.

**Areas of future work in the pipeline:** 
- Run MonoNav on new platforms and in new settings!
- Integrate with ROS for easier interoperability on new systems.
- Integrate improved depth estimation pipelines, which might take in multiple frames or metadata to improve the depth estimate.
- Improve the planner to treat space as unsafe UNTIL explored, which could prevent crashes into occluded obstacles. (The current planner treats unseen space as free.)
- Improve the motion primitive control from open-loop to closed-loop, for more reliable and accurate primitive tracking.

## Citation
```
@inproceedings{simon2023mononav,
  author    = {Nathaniel Simon and Anirudha Majumdar},
  title     = {{MonoNav: MAV Navigation via Monocular Depth Estimation and Reconstruction}},
  booktitle = {Symposium on Experimental Robotics (ISER)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2311.14100}
}
```

