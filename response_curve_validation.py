import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot
class Response_curve(nn.Module):
    def __init__(self,max_power,patch_number):
        #shared parameter
        super(Response_curve,self).__init__()
        self.max_power=max_power
        self.patch_nubmer=patch_number
        tmp_kernel=torch.rand(max_power)
        self.A_array=nn.Parameter(
            tmp_kernel,
            requires_grad=True
        )
    #polynomial function
        tmp_kernel=torch.rand(max_power)
        self.B_array=nn.Parameter(
            tmp_kernel,
            requires_grad=True
        )
        #unique parameter
        tmp_kernel=torch.rand((patch_number,1),dtype=torch.float64)
        self.E_array=nn.Parameter(
            tmp_kernel,
            requires_grad=True
        )
    #exponential function
        #y=a*x^b
        tmp_kernel=torch.tensor([0.57],dtype=torch.float64)
        self.A=nn.Parameter(
            tmp_kernel,
            requires_grad=False
        )
        tmp_kernel=torch.tensor([0.51],dtype=torch.float64)
        self.B=nn.Parameter(
            tmp_kernel,
            requires_grad=False
        )

    def forward(self,X,device):   
        #for example: yi=a2(a1(a0*xi+b0)+b1)+ci  xi=X[i,:]
        #X (patch_number*?)
        X=X.to(device)
        X=self.E_array*X
        # for i in range(self.max_power):
        #     X=self.A_array[i]*X+self.B_array[i]
        return self.A*torch.pow(X,self.B)
    def Get_Et(self,X,device):
        X=X.to(device)
        return self.E_array*X
    def Get_result_from_et(self,Et,device):
        return 
        
def ReadExposure(filename):
    exposure_times=[]
    with open(filename,"r") as f:
        lines=f.readlines()
        for line in lines:
            if(len(line)>6 and line[:7]=="\t<real>"):
                exposure_time=line[7:-8]
                exposure_times.append(float(exposure_time))
    return exposure_times

if __name__=="__main__":

    #get x and y_gt
    photos_root="D:/project/ipad/data/exposure/20230415-15-03-43/"
    exposure_file="exposuretime.txt"
    patch_size=10
    points=np.array([[722,1269],[713,1201],[728,1340],[571,1201],[569,1198],[747,1497],[704,1127]])
    #R 231,187,175,133,94,56,8,
    torch_device=torch.device("cuda:0")
    cpu_device=torch.device("cpu")

#read validation
    if(False):
        x_values=ReadExposure(photos_root+exposure_file)
        photos=glob.glob(photos_root+"Photos/*.png")
        colors=[[],[],[],[],[],[],[]]
        for photoname in photos:
                img=cv2.imread(photoname)
                img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                for i in range(points.shape[0]):
                    begin_y=points[i][1]-patch_size//2
                    end_y=points[i][1]+patch_size//2
                    begin_x=points[i][0]-patch_size//2
                    end_x=points[i][0]+patch_size//2      
                    colors[i].append(np.mean(img[begin_y:end_y,begin_x:end_x,2]).item())
        colors_np=np.array(colors)
        x_values=np.array(x_values)
        i=1
        while(i<x_values.shape[0]):
            if(x_values[i]==x_values[i-1]):
                x_values=np.delete(x_values,i)
                colors_np=np.delete(colors_np,[i],axis=1)
            else:
                i+=1
        np.save(photos_root+"data/colors_validation.npy",colors_np)
        





    x_values=np.load(photos_root+"data/x_values.npy")
    colors=np.load(photos_root+"data/color.npy")
    colors_validation=np.load(photos_root+"data/colors_validation.npy")
    patch_num=colors_validation.shape[0]
    x_values=x_values[10:100]
    colors=colors[:,10:100] 
    colors_validation=colors_validation[:,10:100]
    ############fit response-curve
    x_values=x_values*100
    colors=colors/255
    colors_validation=colors_validation/255
    X=torch.tensor(x_values) #1*?
    Y_gt=torch.tensor(colors_validation) #1*?
    X=X.to(torch_device)
    Y_gt=Y_gt.to(torch_device)


    response_curve=Response_curve(5,patch_num)
    response_curve=response_curve.to(torch_device)

    lr=1e-2
    optimizer=optim.Adam(response_curve.parameters(),lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.8)

    log_folder = photos_root+"optimize_log/"
    os.makedirs(log_folder,exist_ok=True)

    for glob_step in range(5000):
        response_curve.train()
        optimizer.zero_grad()
        Y=response_curve(X,torch_device)
        Y_gt=Y_gt.to(torch_device)
        error=Y-Y_gt
        loss=torch.mean(error*error)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if(glob_step%100==0):
            E_np=response_curve.E_array.detach().cpu().numpy()
            A=response_curve.A.detach().cpu().numpy().item()
            B=response_curve.B.detach().cpu().numpy().item()
            Y_numpy=Y.detach().cpu().numpy()
            for  i in range(patch_num):
                pyplot.plot(x_values,Y_numpy[i,:],label="y={:.2f}*({:.2f}*x)^{:.2f}".format(A,E_np[i].item(),B))
                pyplot.legend()
                pyplot.scatter(x_values,colors_validation[i,:])
                pyplot.savefig(f"{log_folder}{glob_step}.png",bbox_inches="tight")
            pyplot.show()
            print(loss)
            print(torch.mean(error*error,dim=1))





            # response_curve.eval()
            # #draw_point
            # et=response_curve.Get_Et(X,torch_device)
            # et_np=et.to(cpu_device).detach().numpy()
            # Y_gt_np=torch.flatten(Y_gt.to(cpu_device)).detach().numpy()
            # pyplot.scatter(et_np,Y_gt_np)

            # #draw line
            # et_min=np.min(et_np)
            # et_max=np.max(et_np)
            # et_tmp=torch.linspace(et_min,et_max,100)
            # Y_tmp=response_curve.Get_Et(et_tmp,torch_device)
            # pyplot.plot(et_tmp,Y_tmp)
            # #pyplot.show()
            # pyplot.savefig(f"{log_folder}{glob_step}.png")
            # print(loss)







    
    


    