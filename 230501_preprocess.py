#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2
import os
import numpy
import numpy as np
import sys
import argparse
import logging
import pathlib
import json
import threading
import glob
import shutil
from matplotlib import pyplot as plt


# In[16]:


def fix_image_size_(image: numpy.array, expected_pixels: float = 2E6):
    ratio = expected_pixels / (image.shape[0] * image.shape[1])
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image: numpy.array, threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = numpy.var(blur_map)
    return blur_map, score, bool(score < threshold)


def pretty_blur_map(blur_map: numpy.array, sigma: int = 5, min_abs: float = 0.5):
    abs_image = numpy.abs(blur_map).astype(numpy.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = numpy.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)
def mask_invalid_pixel(img,roi):
    x,y,w,h=roi
    img[:y]=0
    img[:,:x]=0
    img[y+h:]=0
    img[:,x+w:]=0
    return img
    


# In[17]:


def find_images(image_paths, img_extensions=['.jpg', '.png', '.jpeg']):
    img_extensions += [i.upper() for i in img_extensions]

    for path in image_paths:
        path = pathlib.Path(path)

        if path.is_file():
            if path.suffix not in img_extensions:
                logging.info(f'{path.suffix} is not an image extension! skipping {path}')
                continue
            else:
                yield path

        if path.is_dir():
            for img_ext in img_extensions:
                yield from path.rglob(f'*{img_ext}')


# In[18]:


def get_result(args,image_path,fix_size,results):
    image = cv2.imread(str(image_path))
    if image is None:
        logging.warning(f'warning! failed to read image from {image_path}; skipping!')
        return 
    #logging.info(f'processing {image_path}')

    if fix_size:
        image = fix_image_size_(image)
    else:
        logging.warning('not normalizing image size for consistent scoring!')

    blur_map, score, blurry = estimate_blur(image, threshold=args.threshold)

    #logging.info(f'image_path: {image_path} score: {score} blurry: {blurry}')
    results.append({'input_path': str(image_path), 'score': score, 'blurry': blurry})


# In[19]:


# with np.load("/home/xiangfeng/NAS/d2ft/playground/230406_motion_database/calib.npz") as camdata:
#     mtx = camdata['mtx']
#     dist = camdata['dist']
#     W = camdata['W']
#     H = camdata['H']
#     print(mtx)
mtx=np.array([ 1.6281047626587153e+03, 0., 5.4239393119148224e+02, 0.,1.6291122927710098e+03, 9.6233223381000607e+02, 0., 0., 1.  ]).reshape((3,3))
dist=np.array([2.7659540439639703e-02, 5.3863103363621501e-01,3.9710184124524791e-03, 8.2175531244021820e-04,-2.3807468377690459e+00]).reshape((5,1))
W=1080
H=1920


# In[20]:


def parse_args():
    parser = argparse.ArgumentParser(description='run blur detection on a single image')
    parser.add_argument('--images_root', type=str, default=r"D:\project\ipad\data\object_with_ipadfrontcamera\20230507-14-30-59")
    parser.add_argument('--threshold', type=float, default=100)
    parser.add_argument('--variable-size', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args, unknown = parser.parse_known_args([])
    args.output_dir = f"{args.images_root}/selected"
    return args

args = parse_args()
fix_size = not args.variable_size
level = logging.DEBUG if args.verbose else logging.INFO
imgpaths = sorted(glob.glob(os.path.join(args.images_root,"*.png")))
print(len(imgpaths))


# In[21]:


results = []
thread_collector = []
max_thread_num = 60
for index, image_path in enumerate(imgpaths):
    print(f"\r{index}/{len(imgpaths)}", end=" ")
    tmp_thread = threading.Thread(target=get_result, args=(args, image_path, fix_size, results))
    tmp_thread.start()
    thread_collector.append(tmp_thread)

    if len(thread_collector) >= max_thread_num:
        for a_thread in thread_collector:
            a_thread.join()
        thread_collector = []
for a_thread in thread_collector:
    a_thread.join()


# In[22]:


scores_ = {}
imgpaths_={}
for result in results:
    idx = int(os.path.basename(result['input_path'])[:-4])
    score = result["score"]
    scores_[idx] = score
    imgpaths_[idx]=result['input_path']
scores = []
imgpaths=[]
for k in sorted(scores_.keys()):
    scores.append(scores_[k])
    imgpaths.append(imgpaths_[k])
assert len(scores) == len(imgpaths)
plt.plot(scores)


# In[34]:


ok_threshold = 5
V = 5
ok_thresholds=np.array([5,10,10,10,5]).reshape((1,V))
n_images = len(imgpaths)
n_frags = n_images // V
n_pieces = 2
n_images // (n_pieces * V)


# In[35]:


def Gamma_correction(img):
    img_=img/255
    
    img_[:,:,0]=np.power(img_[:,:,0],0.53)
    img_[:,:,1]=np.power(img_[:,:,1],0.53)
    img_[:,:,2]=np.power(img_[:,:,2],0.51)
    result=img_*255
    result.astype(np.uint8)
    return result
os.makedirs(f"{args.output_dir}", exist_ok=True)
cnt = 0
left, right = 0, n_pieces * V
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (W,H), 1, (W,H))
while right < n_images:
    s = scores[left:right]
    s = np.array(s).reshape(n_pieces, V)
    #ok_mask = s > ok_threshold
    ok_mask=s>np.repeat(ok_thresholds,n_pieces,0)
    piece_ok = ok_mask.all(axis=-1)
    select_idx = 0
    select_score = -1e8
    for piece_idx in range(n_pieces):
        if not piece_ok[piece_idx]:
            continue
        if s[piece_idx].mean() > select_score:
            select_score = s[piece_idx].mean()
            select_idx = piece_idx
    if select_score > 0:
        base = left + select_idx * V
        select_ids = [idx for idx in range(base, base+V)]
        for idx in select_ids:
            print(scores[idx])
            img = cv2.imread(imgpaths[idx])
            print(os.path.basename(imgpaths[idx]))
            undst = cv2.undistort(img, mtx, dist)
            result= mask_invalid_pixel(undst,roi)
            result=Gamma_correction(result)
            cv2.imwrite(f"{args.output_dir}/{cnt:03d}.png",result)
            print(f"write {args.output_dir}/{cnt:03d}.png")
            cnt += 1
    left += n_pieces * V
    right += n_pieces * V


# In[ ]:





# In[ ]:




