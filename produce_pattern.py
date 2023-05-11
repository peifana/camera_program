import numpy as np
import cv2
screen_size=(2388,1668)
pattern_picture=np.zeros(screen_size).astype(np.uint8)
for i in range(50):
    for j in range(50):
        pattern_picture[i][j]=255
cv2.imwrite("test2.PNG",pattern_picture)
