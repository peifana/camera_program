from dbm import whichdb
import numpy as np
import cv2
import shutil
import argparse
import os
from scipy import optimize
import threading
import glob

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    # print(x)
    y = (Y*data).sum()/total
    # print(y)
    col = data[:, int(y)]
    # print(np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()))
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    # print(width_x)
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default=r"D:/project/ipad/data/screen/20230407/calibrate_screen/photos/")
    parser.add_argument("--fullon_root",default=r"D:/project/ipad/data/screen/20230407/calibrate_screen/mask/")
    parser.add_argument("--calibration_file_root",default=r"D:/project/ipad/data/screen/20230407/calibrate_intrinsic/calibration_result/")
    parser.add_argument("--pattern_num",type=int,default=3468)
    parser.add_argument("--img_patch_size",type=int,default=10)
    args = parser.parse_args()

    tmp_save_root = args.data_root+"tmp/"
    os.makedirs(tmp_save_root,exist_ok=True)

    print("loading intrinsic...")
    s = cv2.FileStorage()
    s.open(args.calibration_file_root+"intrinsic0.yml",cv2.FileStorage_READ)
    cameraA = s.getNode('camera_matrix').mat().astype(np.float64)
    w = int(s.getNode('image_width').real())
    h = int(s.getNode('image_height').real())
    dist = s.getNode('distortion_coefficients').mat()

    if not os.path.exists(args.fullon_root+"mask.png"):
        print("please generate mask.png to:{}".format(args.fullon_root+"mask.png"))
        input()
    mask = cv2.imread(args.fullon_root+"mask.png")
    mask = cv2.undistort(mask,cameraA,dist)
    mask = np.where((mask>0).any(axis=2,keepdims=True),np.ones_like(mask),np.zeros_like(mask))
    print("mask.shape:")
    print(mask.shape)
    valid_y,valid_x = np.where(mask[:,:,0] > 0)
    y_min = valid_y.min()
    y_max = valid_y.max()
    x_min = valid_x.min()
    x_max = valid_x.max()
    valid_width = x_max-x_min+1
    valid_height = y_max-y_min+1
    print("{} {} {} {} {} {}".format(x_min,y_min,x_max,y_max,valid_height,valid_width))

    img_patch_size = args.img_patch_size
    total_img_num = args.pattern_num
    center_collector = [None]*total_img_num
    invalid_flag = [False]*total_img_num
    mask = cv2.imread(args.fullon_root+"mask.png")
    def find_center(data_root,which_img,cameraA,dist,y_min,valid_height,x_min,valid_width,img_patch_size,center_collector,invalid_flag):
        try:
            img = cv2.imread(data_root+"{:0>4d}.png".format(which_img))
            print("{:0>4d}.png".format(which_img))
            img = (mask == 255) * img
            img = cv2.undistort(img,cameraA,dist)
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_gray = cv2.medianBlur(img_gray,3)
            cv2.imwrite("test.png",img_gray)

            img_gray_sub = img_gray[y_min:y_min+valid_height,x_min:x_min+valid_width]
            maxvalue=np.max(img_gray_sub)

            shining_pidxels = np.stack(np.where(img_gray_sub > 150),axis=1)
            cX = int(np.median(shining_pidxels[:,1]))+x_min
            cY = int(np.median(shining_pidxels[:,0]))+y_min
            start_X = cX-img_patch_size//2
            start_Y = cY-img_patch_size//2

            img_patch = img_gray[start_Y:start_Y+img_patch_size,start_X:start_X+img_patch_size]

            p = fitgaussian(img_patch)
            cX_fitted = p[1]+start_X
            cY_fitted = p[2]+start_Y
            # cX=int(cX_fitted)
            # cY=int(cY_fitted)
            # radius=3
            # while(1):
            #     next=np.array([[0,0],[-1,0],[1,0],[0,-1],[0,1]],dtype=np.int32)
            #     total=np.zeros(5)
            #     for i in range(5):
            #         tempx=cX+next[i][1]
            #         tempy=cY+next[i][0]
            #         total[i]=np.sum(img_gray[tempy-radius:tempy+radius,tempx-radius:tempx+radius])
            #     index=np.where(total==np.max(total))[0].item()
            #     if(index==0):
            #         break
            #     else:
            #         cX=cX+next[index][1]
            #         cY=cY+next[index][0]


            print(f"X:{cX_fitted},Y:{cY_fitted}")

            fit = gaussian(*p)
            img_patch_fitted = fit(*np.indices(img_patch.shape))
            img_fitted = np.zeros_like(img_gray)
            img_fitted[start_Y:start_Y+img_patch_size,start_X:start_X+img_patch_size] = img_patch_fitted

            center_collector[which_img]=np.array([cX_fitted,cY_fitted])



            log_img = np.stack([img_gray,np.zeros_like(img_gray),img_fitted],axis=2)
            log_patch = np.stack([img_patch,np.zeros_like(img_patch),img_patch_fitted],axis=2)
            img = cv2.imread(data_root+"{:0>4d}.png".format(which_img))
            cv2.circle(img,(int(cX_fitted),int(cY_fitted)),5,(255,0,0))
            cv2.imwrite(tmp_save_root+"{:0>4d}.png".format(which_img),img)
        except Exception as e:
            print("error in {}".format(which_img))
            print(e)
            center_collector[which_img] = np.array((-1,-1))
            invalid_flag[which_img] = True




    thread_collector = []
    max_thread_num = 30
    for which_img in range(total_img_num):
        print("{}/{}".format(which_img,total_img_num))
        
        tmp_thread = threading.Thread(target=find_center,args=(args.data_root,which_img,cameraA,dist,y_min,valid_height,x_min,valid_width,img_patch_size,center_collector,invalid_flag))
        tmp_thread.start()
        thread_collector.append(tmp_thread)

        if len(thread_collector) >= max_thread_num:
            for a_thread in thread_collector:
                a_thread.join()
            thread_collector = []
    for a_thread in thread_collector:
        a_thread.join()


    center_collector = np.stack(center_collector,axis=0)
    np.savetxt(args.data_root+"center_coords.csv",center_collector,delimiter=',')
    with open(args.data_root+"invalid_id.txt","w") as pf:
        [pf.write("{}\n".format(tmp_id)) for tmp_id,flag_value in enumerate(invalid_flag) if flag_value]

    # for which_img in range(total_img_num):
    #     find_center(args.data_root,which_img,cameraA,dist,y_min,valid_height,x_min,valid_width,img_patch_size,center_collector,invalid_flag)
    # find_center(args.data_root,0,cameraA,dist,y_min,valid_height,x_min,valid_width,img_patch_size,center_collector,invalid_flag)


