#!/usr/bin/env python

'''
ADAPTED FROM OPENCV SAMPLES
https://github.com/opencv/opencv/blob/4.x/samples/python/calibrate.py
SAMPLE CALL:
python calibrate.py --debug ./calibration_output -w 6 -h 8 -t chessboard --square_size=35 ./calibration_pictures/frame*.jpg


SEE INSTRUCTIONS FROM OPENCV:
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [-w <width>] [-h <height>] [-t <pattern type>] [--square_size=<square size>] [<image mask>]

usage example:
    calibrate.py -w 4 -h 6 -t chessboard --square_size=50 ../data/left*.jpg

default values:
    --debug:    ./output/
    -w: 4
    -h: 6
    -t: chessboard
    --square_size: 10
    --marker_size: 5
    --threads: 4

NOTE: Chessboard size is defined in inner corners. Charuco board size is defined in units, and has been removed from this sample.
'''

import numpy as np
import cv2 as cv
import json
import os

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def main():
    import sys
    import getopt

    args, img_names = getopt.getopt(sys.argv[1:], 'w:h:t:', ['debug=','square_size=', 'threads=', ])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('-w', 4)
    args.setdefault('-h', 6)
    args.setdefault('-t', 'chessboard')
    args.setdefault('--square_size', 10)
    args.setdefault('--threads', 4)

    assert img_names, 'Did you provide a path for images?'

    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)

    height = int(args.get('-h'))
    width = int(args.get('-w'))
    pattern_type = str(args.get('-t'))
    square_size = float(args.get('--square_size'))

    pattern_size = (width, height)
    if pattern_type == 'chessboard':
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]

    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found = False
        corners = 0
        if pattern_type == 'chessboard':
            found, corners = cv.findChessboardCorners(img, pattern_size)
            if found:
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
                frame_img_points = corners.reshape(-1, 2)
                frame_obj_points = pattern_points
        else:
            print("unknown pattern type", pattern_type)
            return None

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            if pattern_type == 'chessboard':
                cv.drawChessboardCorners(vis, pattern_size, corners, found)
            _path, name, _ext = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_board.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('pattern not found')
            return None

        print('           %s... OK' % fn)
        return (frame_img_points, frame_obj_points)

    threads_num = int(args.get('--threads'))
    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names)

    chessboards = [x for x in chessboards if x is not None]
    for idx, (corners, pattern_points) in enumerate(chessboards):
        if len(corners) < 4:
            print("Not enough obj/img points for %d, skipping image!" % idx)
        else:
            img_points.append(corners)
            obj_points.append(pattern_points)
    
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
    print("newcameramtx:\n", newcameramtx)
    print("roi: ", roi)


    data = {
    "RMS": rms,
    "CameraMatrix": camera_matrix.tolist(),
    "DistortionCoefficients": dist_coefs.ravel().tolist(),
    "NewCameraMatrix": newcameramtx.tolist(),
    "ROI": roi
    }
    input_dirname = img_names[0].split('/')[1].split('.')[0]
    if input_dirname == 'transform_output':
        file_path = "check_intrinsics.json"
    else:
        file_path = "intrinsics.json"

    with open(file_path, "w") as json_file:
        json.dump(data, json_file)

    # undistort the image with the calibration
    print('')
    for fn in img_names if debug_dir else []:
        _path, name, _ext = splitfn(fn)
        # img_found = os.path.join(debug_dir, name + '_board.png')
        outfile = os.path.join(debug_dir, name + '_undistorted.png')

        img = cv.imread(fn)
        if img is None:
            continue

        dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # # crop and save the image
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

