import torch
import numpy as np
import  argparse
import torch.optim as optim
from torch import nn
class Function(nn.module):
        def init(self,args):
            super(Function,self).__init__()

            tmp_kernel=torch.tensor([args.liner_parameter],dtype=torch.float32)
            self.liner_parameter=nn.Parameter(
                data=tmp_kernel,
                requires_grad=True
            )

            tmp_kernel=torch.tensor([args.pow_parameter],dtype=torch.float32)
            self.pow_parameter=nn.Parameter(
                data=tmp_kernel,
                requires_grad=True
            )

            tmp_kernel=torch.tensor([args.overexposure_parameter],dtype=torch.float32)
            self.overexposure_parameter=nn.Parameter(
                data=tmp_kernel,
                requires_grad=True
            )

            tmp_kernel=torch.tensor([args.boundary],dtype=torch.int32)
            self.boundary=nn.Parameter(
                data=tmp_kernel,
                requires_grad=True
            )
        def forward(self,x_array):
             y1=self.liner_parameter[0]*x_array+self.liner_parameter[1]
             y2=self.pow_parameter[0]*x_array+self.pow_parameter[1]
             y2=torch.pow(y2,self.pow_parameter[2])
             y3=self.overexposure_parameter[0]*x_array+self.overexposure_parameter[1]
             result=torch.cat(y1[x_array<self.boundary[0]],y2[x_array>=self.boundary[0] and x_array<self.boundary[1]],y3[x_array>self.boundary[1]])
             return result
        
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add("--liner_parameter",default=(0,0))#y=a1*x+a2
    parser.add("--pow_parameter",default=(0,0,0))#y=pow(a1*x+a2,a3)
    parser.add("--overexposure_parameter",default=(0,0))#y=a1*x+a2
    parser.add("--boundary",default=(0,1.0))
