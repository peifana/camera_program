import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.optimize import curve_fit
import os
import struct
def ReadExposure(filename):
    exposure_times=[]
    with open(filename,"r") as f:
        lines=f.readlines()
        for line in lines:
            if(len(line)>6 and line[:7]=="\t<real>"):
                exposure_time=line[7:-8]
                exposure_times.append(float(exposure_time))
    return exposure_times
def test(x,a1,b1,a2,b2,p2,c2):
    #return np.piecewise(x,[x<x1,x>=x1 and x<=x2,x<x2],[lambda x:a1*x+b1,lambda x:np.power(a3*x+b3,p3),lambda x:a2*x+b2])
    return np.piecewise(x,[x<0.01,x>=0.01],[lambda x:a1*x+b1,lambda x:np.power(a2*x+b2,p2)+c2])
    #return np.power(a1*x+b1,p1)
if __name__=="__main__":
    patch_size=5
    picture_num=50
    exposure_file="exposuretime.txt"
       #from white to the black
    #points=np.array([[910,1290],[910,1200],[910,1110],[910,1025],[910,930],[910,840]])# 040816
    #points=np.array([[834,1494],[829,1410],[824,1326],[827,1240],[826,1155],[831,1062]])
    #points=np.array([[921,1553],[921,1467],[920,1381],[920,1292],[918,1205],[917,1112]])
    #points=np.array([[919,1626],[919,1549],[913,1467],[910,1390],[905,1314],[906,1227]])
    #points=np.array([[995,1558],[995,1473],[996,1388],[992,1303],[993,1215],[994,1128]]) #200  20230410-12-17-03
    #points=np.array([[962,1451],[959,1376],[958,1300],[956,1220],[956,1141],[953,1055]])  #150 20230410-12-25-03
    #points=np.array([[946,1261],[947,1340],[945,1416],[944,1493],[940,1565],[941,1638]])  #nature 20230412-14-27-15
    #points=np.array([[934,1538],[932,1456],[930,1375],[927,1295],[924,1214],[925,1133]])  #nature 20230412-14-27-15
    #points=np.array([[910,1528],[909,1441],[908,1358],[907,1277],[905,1196],[902,1113]])  #nature 20230412-14-27-15
    #points=np.array([[804,1527],[808,1472],[815,1418],[818,1363],[826,1309],[833,1249]])
    points=np.array([[897,1121],[900,1177],[900,1238],[900,1302],[905,1364],[906,1432]])


    
    photos_root=r"D:\project\ipad\data\exposure\20230503-15-36-44"
    save_root=os.path.join(photos_root,"result")
    os.makedirs(save_root,exist_ok=True)
    #if not os.path.exists(photos_root+"data/color.npy"):
    if True:
        x_values=[i for i in range(50)]
        colors=[]
        for i in range(picture_num):
            temp_color=[]
            path=os.path.join(photos_root,f"y_{i}")
            # with open(path,'rb') as file:
            #     Y_datas=[]
            #     size=os.path.getsize(path)
            #     for j in range(size):
            #         data=file.read(1)
            #         num=struct.unpack('B', data)
            #         Y_datas.append(num[0])
            #     img=np.resize(np.array(Y_datas,dtype=np.uint8),(1920,size//1920))
            #     cv2.imshow("img",img)
            #     cv2.waitKey(1)
            img=cv2.imread(os.path.join(photos_root,f"{i}.png"))
            img=img[:,:,0]
            for j in range(6):
                begin_y=points[j][1]-patch_size//2
                end_y=points[j][1]+patch_size//2
                begin_x=points[j][0]-patch_size//2
                end_x=points[j][0]+patch_size//2 
                temp_color.append(np.mean(img[begin_y:end_y,begin_x:end_x]).item())
            colors.append(temp_color)
        x_values=np.array(x_values)
        np.save(os.path.join(save_root,"x_values.npy"),x_values)
        colors=np.array(colors)
        np.save(os.path.join(save_root,"colors_Bbuffer.npy"),colors)
    
    colors=np.load(os.path.join(save_root,"colors_Bbuffer.npy"))
    x_values=np.load(os.path.join(save_root,"x_values.npy"))
    colors=colors[:,:]
    x_values=x_values[:]
    plt.plot(x_values,colors)
    plt.show()

        


    #     i=1
    #     while(i<len(x_values) and i<len(colors)):
    #         if x_values[i] ==x_values[i-1]:
    #             del x_values[i]
    #             del colors[i]
    #         else:
    #             i=i+1
    #     colors_np=np.array(colors)
    #     colors_np=colors_np/255
    #     np.save(photos_root+"data/color.npy",colors_np)
    #     np.save(photos_root+"data/x_values.npy",np.array(x_values))

    # colors_R_reference=[243,200,160,122,85,52]
    # x_values=np.load(photos_root+"data/x_values.npy")
    # colors=np.load(photos_root+"data/color.npy")
            
    # #colors=np.power(colors,2.2)
    # for i in range(6):
    #     color_selected=colors[i,:]
    #     x_values_selected=x_values[:]
    #     if i<1:
    #         param, param_cov = curve_fit(test, x_values, color_selected)
    #         print(f"param :{param}")
    #         print(f"param_cov{param_cov}")

    #         y=test(x_values,*param)
    #         plt.plot(x_values,y)
    #     plt.plot(x_values_selected,color_selected)
    # # for i in range()
    # plt.savefig(photos_root+"data/result.png")
    # plt.show()


    