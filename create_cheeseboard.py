import cv2
import numpy as np
x_number=12
y_nubmer=8
box_length=80
width=x_number*box_length
height=y_nubmer*box_length
chessBoard=np.zeros((height,width),dtype=np.uint8)
chessBoard[:]=255
for i_y in range(y_nubmer):
      for i_x in range(x_number):
            if(i_y%2== i_x%2):
                  chessBoard[i_y*box_length:i_y*box_length+box_length,i_x*box_length:i_x*box_length+box_length]=0
cv2.imwrite("chessBoard.png",chessBoard)
