from tabnanny import verbose
import numpy as np
import cv2
import os
import sys
import torch
import torch.optim as optim
import argparse
from scipy.spatial.transform import Rotation as R
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_data_root",default=r"D:/project/ipad/data/screen/20230406/calibrate_intrinsic/calibration_result/")
    parser.add_argument("--save_path",default=r"D:/project/ipad/data/screen/20230406/result/")
    parser.add_argument("--light_center_path",default=r"D:/project/ipad/data/screen/20230406/calibrate_screen/Photos/")

    parser.add_argument("--pixel_length",default=(0.096212121,0.096212121)) #mm
    parser.add_argument("--size_in_pixel",default=(40,40))
    parser.add_argument("--step_in_pixel",default=(80,80)) 
    parser.add_argument("--begin_in_pixel",default=(20,20))                 
    parser.add_argument("--sphere_radius",type=np.double,default=100.0)
    parser.add_argument("--base_pose",default=(107,0,0))
    parser.add_argument("--x_direction",default=(-1,0,0))
    parser.add_argument("--screen_resolution",default=(2048,2732))
    parser.add_argument("--pattern_num",default=884)

    args = parser.parse_args()
    torch_device = torch.device("cuda:0")
    screen_resolution=[int(a) for a in args.screen_resolution]
    pattern_num=int(args.pattern_num)

    ## -----calibrated camera intrinsic
    print("loading intrinsic...")
    s = cv2.FileStorage()
    s.open(args.calibration_data_root+"intrinsic0.yml", cv2.FileStorage_READ)
    cameraA = s.getNode('camera_matrix').mat().astype(np.float64)
    cam_A_matrix_tc = torch.from_numpy(cameraA).to(torch_device)
    w = int(s.getNode('image_width').real())
    h = int(s.getNode('image_height').real())
    dist = s.getNode('distortion_coefficients').mat()

    ## -----calibrated camera extrinsic
    print("loading camera pos...")
    point_collector = []
    color_colelctor = []

    rvecs = np.fromfile(args.calibration_data_root+"rvecs.bin",np.float32).reshape((-1,3)).astype(np.float64)
    tvecs = np.fromfile(args.calibration_data_root+"tvecs.bin",np.float32).reshape((-1,3,1)).astype(np.float64)
    for which_cam in range(rvecs.shape[0]):
        RMatrix,_ = cv2.Rodrigues(rvecs[which_cam])
        cam_pos = -np.matmul(RMatrix.T,tvecs[which_cam])
        point_collector.append(cam_pos)
        color_colelctor.append(np.ones((3,),np.float32))
        # print(cam_pos)

    rvec = np.fromfile(args.calibration_data_root+"rvec_plane.bin",np.float32).astype(np.float64)
    tvec = np.fromfile(args.calibration_data_root+"tvec_plane.bin",np.float32).reshape((3,1)).astype(np.float64)
    print(f"rvec:{rvec}")
    print(f"tvec:{tvec}")
    RMatrix,_ = cv2.Rodrigues(rvec)
    # cam_R_matrix_tc = torch.from_numpy(RMatrix).to(torch_device)
    # cam_T_vec_tc = torch.from_numpy(tvec).to(torch_device)
    # cam_pos_plane = -np.matmul(RMatrix.T,tvec)
    origin=tvec.squeeze()
    print(f"origin: {origin.shape}")
    print(f"RMatrix:{RMatrix}")
    norm_plane=np.matmul(RMatrix,np.array([0,0,1],dtype=np.float64))
    print(f"norm_plane:{norm_plane.shape}")
    plane_eq=np.array([norm_plane[0],norm_plane[1],norm_plane[2],-np.dot(norm_plane,origin)]) #4*1
    #plane_eq=np.array([0,0,-1,-200],dtype=np.float64)
    print(f"plane_eq: {plane_eq}")

    ## -----reflected light center in image
    print("loading light centers....")
    light_centers = np.loadtxt(args.light_center_path+"center_coords.csv",dtype=np.float32,delimiter=',').astype(np.float64)
    light_centers_tc = torch.from_numpy(light_centers).to(torch_device)
    try:
        with open(args.light_center_path+"invalid_id.txt","r") as pf:
            invalid_light_ids = np.array([int(a) for a in pf.read().strip("\n").split("\n")])
        with_invalid_light = True
    except Exception as e:
        with_invalid_light = False
    ## -----build a standard ls setup
    #ls = lightstage(setup_args)

    ############################################
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class ls_setup_net(nn.Module):
        def __init__(self,args):
            super(ls_setup_net,self).__init__()

            tmp_kernel = torch.from_numpy(np.array([float(a) for a in args.base_pose[:2]]))
            #tmp_kernel = torch.from_numpy(np.fromfile(args.calibration_data_root+"calibration_result/base_light_pos.bin",np.float32).reshape((5, 4, 3)).astype(np.float64))
            self.base_pose_xy = nn.Parameter(#(grp_num,3)
                data = tmp_kernel[:2],
                requires_grad=True
            )
            tmp_kernel=torch.tensor(float(args.base_pose[2]),dtype=torch.float64).view(1)
            self.base_pose_z=nn.Parameter(
                data=tmp_kernel,
                requires_grad=False
            )
            #tmp_kernel = torch.from_numpy(np.fromfile(args.calibration_data_root+"calibration_result/face_norm.bin",np.float32).reshape((5, 3)).astype(np.float64))
            #tmp_kernel = torch.from_numpy(ls.face_norm.astype(np.float64))
            # print("face_norm:")
            # print(tmp_kernel)
            tmp_kernel=torch.tensor([float(a) for a  in args.x_direction],dtype=torch.float64)
            self.x_direction = nn.Parameter(#(grp_num,2)
                data = tmp_kernel,
                requires_grad=False
            )
            tmp_kernel=torch.tensor([float(args.pixel_length[i]) for i in range(2) ],dtype=torch.float64)
            self.pixel_length=nn.Parameter(
                data=tmp_kernel,
                requires_grad=False
            )
            tmp_kernel=torch.from_numpy(plane_eq[:3])
            #tmp_kernel=torch.from_numpy(no.fromfile())
            self.plane_eq_abc=nn.Parameter(
                data=tmp_kernel,
                requires_grad=False
            )

            tmp_kernel=torch.tensor(plane_eq[3],dtype=torch.float64).view(1)
            #tmp_kernel=torch.from_numpy(no.fromfile())
            self.plane_eq_d=nn.Parameter(
                data=tmp_kernel,
                requires_grad=True
            )


            # tmp_kernel = torch.from_numpy(np.fromfile(args.calibration_data_root+"calibration_result/sphere_center.bin",np.float32).astype(np.float64))
            # print("sphere_center:")
            # print(tmp_kernel)
            # self.sphere_center = nn.Parameter(#(3,)
            #     data = tmp_kernel,
            #     requires_grad=True
            # )

            # tmp_kernel = torch.tensor([98.62118872486057], dtype=torch.float64)
            # # print("sphere_r")
            # # print(tmp_kernel)
            # self.sphere_radius = nn.Parameter(#(3,)
            #     data = tmp_kernel,
            #     requires_grad=False
            # )
           



        def get_pos(self,device,base_pose,args):
            x_direction= self.x_direction
            x_direction=x_direction/torch.linalg.norm(x_direction)
            z_direction=torch.tensor([0,0,-1],dtype=torch.float64).to(device)
            y_direction=torch.cross(z_direction,x_direction)

            x_step=int(args.step_in_pixel[0])
            y_step=int(args.step_in_pixel[1])
            x_begin_in_pixel=int(args.begin_in_pixel[0])
            y_begin_in_pixel=int(args.begin_in_pixel[1])
            x_sample_size=(int(args.screen_resolution[0])-int(args.size_in_pixel[0]))//int(args.step_in_pixel[0])+1
            y_sample_size=(int(args.screen_resolution[1])-int(args.size_in_pixel[1]))//int(args.step_in_pixel[1])+1
            light_pos = []
            
            for i in range(y_sample_size):
                for j   in range(x_sample_size):
                    offset_pixel=torch.tensor([y_begin_in_pixel+j*x_step,x_begin_in_pixel+i*y_step]).to(device)
                    offset_real=offset_pixel*self.pixel_length
                    temp_pos=base_pose+offset_real[0]*x_direction+offset_real[1]*y_direction
                    light_pos.append(temp_pos)
                
            light_pos = torch.stack(light_pos, dim = 0)
            #print(light_pos.shape)
            #print("----get pose done----")

            return light_pos.to(device)

        def forward(self, device,cam_A_matrix,used_light_id_list):
            def Find_symmetry_point(poses,face_eq): #n*3,    4,
                
                face_eq_matrix=torch.reshape(face_eq,(4,1))
                square=torch.sum(torch.square(face_eq_matrix[:3,0]))
                t=-2*(poses*face_eq_matrix[:3,0]+face_eq[3])/square*torch.transpose(face_eq_matrix[:3],0,1)
                result=poses+t
                return result



            # print("*******forward begin*****")
            base_pose=torch.cat((self.base_pose_xy,self.base_pose_z))
            plane_eq=torch.cat((self.plane_eq_abc,self.plane_eq_d))
            light_poses = self.get_pos(device,base_pose,args)
            light_poses = light_poses[used_light_id_list]
            light_num = light_poses.shape[0]
            N=Find_symmetry_point(light_poses,plane_eq)

            #step.3 project reflectance point to image
            projected_point = torch.matmul(cam_A_matrix[None,:,:],N[:,:,None])
            projected_point = torch.squeeze(projected_point,dim=2)
            projected_point = projected_point[:,:2] / projected_point[:,[2]]
            # print("******forward done******")

            return projected_point,N


    lightstage = ls_setup_net(args)
    lightstage.to(torch_device)

    lr = 1e-1# if train_configs["training_mode"] == "pretrain" else 1e-5
    #lr=1e-5
    optimizer = optim.Adam(lightstage.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.8)
    l2_loss_fn = torch.nn.MSELoss(reduction='mean')

    log_folder = args.calibration_data_root+"optimize_light_pos_log/"
    os.makedirs(log_folder,exist_ok=True)

    for global_step in range(5000):
        lightstage.train()
        optimizer.zero_grad()
        projected_uv_collector = []
        N_collector = []
        
        batch_size = 3468

        start_time = time.time()
        # for now_batch in range(3468 // batch_size):
        #     #print(str(now_batch * batch_size) + "/" + str(3468))
        #     projected_uv,N = lightstage(torch_device,cam_A_matrix_tc,
        #     np.arange(now_batch * batch_size, (now_batch + 1) * batch_size))
        #     projected_uv_collector.append(projected_uv)
        #     N_collector.append(N)
        # # print(projected_uv)
        projected_uv,N = lightstage(torch_device,cam_A_matrix_tc,
             np.arange(pattern_num))
        projected_uv_collector.append(projected_uv)
        N_collector.append(N)
        print("cost time:{}s".format(time.time()-start_time))
        
        projected_uv_collector = torch.cat(projected_uv_collector,dim=0)
        N_collector = torch.cat(N_collector,dim=0)

        if global_step % 10 == 0:

            projected_uv_collector_np = projected_uv_collector.detach().cpu().numpy()
            tmp_img = np.ones((h,w,3),np.uint8) * 255
            with torch.no_grad():
                base_pose=torch.cat((lightstage.base_pose_xy,lightstage.base_pose_z))
                light_pos = lightstage.get_pos(torch_device,base_pose,args)
            for which_pixel in range(projected_uv_collector_np.shape[0]):
                x,y = int(projected_uv_collector_np[which_pixel,0]),int(projected_uv_collector_np[which_pixel,1])
                tmp_color = (0,0,255)
                cv2.circle(tmp_img,(x,y),2,tmp_color,-1)
                #cv2.circle(tmp_img,(x,y),5,(0,0,int(which_pixel*0.28)),-1)

                x,y = int(light_centers[which_pixel,0]),int(light_centers[which_pixel,1])
                tmp_color = (255,0,0)
                #cv2.circle(tmp_img,(x,y),5,(int(which_pixel*0.28),0,0),-1)
                cv2.circle(tmp_img,(x,y),2,tmp_color,-1)
            cv2.imwrite(log_folder+"proj_{}.png".format(global_step),tmp_img[:,:,::-1])

            lightstage.base_pose_xy.detach().cpu().numpy().astype(np.float32).tofile(args.save_path+"base_pose_xy.bin")
            lightstage.x_direction.detach().cpu().numpy().astype(np.float32).tofile(args.save_path+"x_direction.bin")
            lightstage.pixel_length.detach().cpu().numpy().astype(np.float32).tofile(args.save_path+"pixel.bin")
            lightstage.plane_eq_abc.detach().cpu().numpy().astype(np.float32).tofile(args.save_path+"plane_eq_abc.bin")
            lightstage.plane_eq_d.detach().cpu().numpy().astype(np.float32).tofile(args.save_path+"plane_eq_d.bin")
            s = cv2.FileStorage(args.save_path+"screen.yml", cv2.FileStorage_WRITE)
            s.write('base_pose_xy',lightstage.base_pose_xy.detach().cpu().numpy().astype(np.float32))
            s.write('x_direction', lightstage.x_direction.detach().cpu().numpy().astype(np.float32))
            s.write('pixel_length', lightstage.pixel_length.detach().cpu().numpy().astype(np.float32))


        uv_error = projected_uv_collector-light_centers_tc#[:1024]#[np.array([0, 15, 479, 494]) + 16]

        if with_invalid_light:
            uv_error[invalid_light_ids] = 0.0
        print("max_error:")
        print(torch.argmax(torch.sum(uv_error**2, dim = 1)), torch.max(torch.sum(uv_error**2, dim = 1)))
        uv_loss = torch.mean(uv_error*uv_error)
        base_loss=torch.mean(uv_error[0]*uv_error[0])
        LAMBDA=1
        total_loss=uv_loss+LAMBDA*base_loss
        print("gloabal step:{} uv_loss:{:.2f}   base_pose_xy:{}".format(global_step,uv_loss.item(),lightstage.base_pose_xy.detach().cpu().numpy()))
        total_loss.backward()
        optimizer.step()
        scheduler.step(uv_loss)