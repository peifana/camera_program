#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import shutil
import argparse
import time
import hashlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root")
    parser.add_argument("--calibration_result_root")
    parser.add_argument("--chessboard_corner_num")#9,21
    parser.add_argument("--chessboard_block_edge_size",type=float)
    parser.add_argument("--task",choices=["add_intrinsic_images","add_extrinsic_image","calibrate"])

    args = parser.parse_args()

    selected_images_root = args.data_root.strip("/")+"_selected/"
    if not os.path.exists(selected_images_root):
        os.makedirs(selected_images_root)

    # Defining the dimensions of checkerboard
    CHECKERBOARD = [int(a) for a in args.chessboard_corner_num.split(',')]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)[:,::-1]*args.chessboard_block_edge_size
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    if args.task == "add_intrinsic_images":
        print("************************************")
        print(args.data_root)
        images = glob.glob('{}*.png'.format(args.data_root))
        images+=glob.glob('{}*.jpeg'.format(args.data_root))
        print(len(images))
    elif args.task == "calibrate":
        images = glob.glob('{}*.png'.format(selected_images_root))
        images+=glob.glob('{}*.jpeg'.format(args.data_root))
        plane_img_name=images[0]
        for a_name in images:
            if "plane.png" in a_name:
                plane_img_name = a_name
                break
        images.remove(plane_img_name)
        images.insert(0,plane_img_name)
    elif args.task == "add_extrinsic_image":
        images = [args.data_root+"0000.png"]
    counter = 0
    plane_img_id = counter
    for fname in images:
        identity_name = hashlib.sha256(bytes(fname+time.ctime(os.path.getmtime(fname)),'utf-8')).hexdigest() if args.task == "add_intrinsic_images" else "plane"
        if args.task == "add_intrinsic_images" and os.path.exists(selected_images_root+"{}.png".format(identity_name)):
                print("file:{} has already been added to intrinsic sets".format(fname))
                continue
        if args.task == "add_extrinsic_image" and os.path.exists(selected_images_root+"{}.png".format(identity_name)):
                print("extrinsic image has already been added, previous one will be replaced.")
        img = cv2.imread(fname)
        #gray=img
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        #ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        print(gray.shape)
        print(CHECKERBOARD)
        ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD[0],CHECKERBOARD[1]), None)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)
            

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (CHECKERBOARD[0],CHECKERBOARD[1]), corners2, ret)
            # if not (args.task == "add_intrinsic_images" or args.task == "add_extrinsic_image"):
                # for i,tmp_corner in enumerate(corners):
                #     cv2.putText(img, "({:.0f},{:.0f})".format(objp[0,i,0],objp[0,i,1]), tmp_corner.astype(np.int32)[0], cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        
            #cv2.imshow('img',cv2.resize(img,(img.shape[1]//5,img.shape[0]//5)))
            if (args.task == "add_intrinsic_images" or args.task == "add_extrinsic_image") or (args.task == "calibrate" and "plane" not in fname):
                cv2.waitKey(1)
            else:
                cv2.waitKey(0)

            if (args.task == "add_intrinsic_images" or args.task == "add_extrinsic_image"):
                shutil.copyfile(fname,selected_images_root+"{}.png".format(identity_name))
            if "plane" in fname:
                plane_img_id = counter
            counter+=1
            print("successfully added file:{} {}/{}".format(fname,counter,len(images)))
        else:
            if args.task == "add_intrinsic_images" or args.task == "add_extrinsic_image":
                print("failed to find chess corners in:{}".format(fname))
                os.remove(fname)
            elif args.task == "calibrate":
                print("ERROR occured! The image:{} is not added by me.".format(fname))
        print(identity_name)
    if args.task == "add_intrinsic_images" or args.task == "add_extrinsic_image":
        exit()
    print(counter)
    cv2.destroyAllWindows()
    
    h,w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    if not os.path.exists(args.calibration_result_root):
        os.makedirs(args.calibration_result_root)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    rvecs = np.stack(rvecs,axis=0)
    tvecs = np.stack(tvecs,axis=0)
    mtx.astype(np.float32).tofile(args.calibration_result_root+"mtx.bin")
    dist.astype(np.float32).tofile(args.calibration_result_root+"dist.bin")
    rvecs.astype(np.float32).tofile(args.calibration_result_root+"rvecs.bin")
    tvecs.astype(np.float32).tofile(args.calibration_result_root+"tvecs.bin")
    rvecs[plane_img_id].astype(np.float32).tofile(args.calibration_result_root+"rvec_plane.bin")
    tvecs[plane_img_id].astype(np.float32).tofile(args.calibration_result_root+"tvec_plane.bin")
    # print("Camera matrix {}: \n".format(mtx.shape))
    # print(mtx)
    # print("dist {}: \n".format(dist.shape))
    # print(dist)
    # print("rvecs {}: \n".format(rvecs.shape))
    # print(rvecs)
    # print("tvecs {}: \n".format(tvecs.shape))
    # print(tvecs)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "mean error: {}".format(mean_error/len(objpoints)) )

    s = cv2.FileStorage(args.calibration_result_root+"intrinsic0.yml", cv2.FileStorage_WRITE)
    nframes: 271
    s.write('image_width', w)
    s.write('image_height', h)
    s.write('board_width', CHECKERBOARD[1])
    s.write('board_height', CHECKERBOARD[0])
    s.write('camera_matrix', mtx)
    s.write('distortion_coefficients', dist.T)
    s.write('avg_reprojection_error', mean_error/len(objpoints))