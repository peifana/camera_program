import argparse
import cv2
import numpy as np
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern_root")
    parser.add_argument("--task",choices=["intrinsic","light_pos","light_pos_fullon","light_pos_fulloff","scan", "test", "calibrate_light"])
    parser.add_argument("--image_num",type=int)
    parser.add_argument("--edge_length",type=int,default=128)

    args = parser.parse_args()

    if not os.path.exists(args.pattern_root):
        os.makedirs(args.pattern_root)

    if args.task == "intrinsic":
        for i in range(args.image_num):
            blank_pattern = np.zeros((args.edge_length * 4, args.edge_length * 3, 3),np.uint8)
            blank_pattern[0:args.edge_length, args.edge_length:args.edge_length * 2, :] = 255
            blank_pattern[args.edge_length:args.edge_length * 2, :, :] = 255
            blank_pattern[args.edge_length * 2:args.edge_length * 4, args.edge_length:args.edge_length * 2, :] = 255
            cv2.imwrite(args.pattern_root+"{}.png".format(i),blank_pattern)
    
    if args.task == "light_pos_fullon":
        for i in range(args.image_num):
            blank_pattern = np.zeros((args.edge_length * 4, args.edge_length * 3, 3),np.uint8)
            blank_pattern[0:args.edge_length, args.edge_length:args.edge_length * 2, :] = 255
            blank_pattern[args.edge_length:args.edge_length * 2, :, :] = 255
            blank_pattern[args.edge_length * 2:args.edge_length * 4, args.edge_length:args.edge_length * 2, :] = 255
            cv2.imwrite(args.pattern_root+"{}.png".format(i),blank_pattern)
        
    if args.task == "light_pos_fulloff":
        for i in range(args.image_num):
            blank_pattern = np.zeros((args.edge_length * 4, args.edge_length * 3, 3),np.uint8)
            cv2.imwrite(args.pattern_root+"{}.png".format(i),blank_pattern)

    if args.task == "scan":
        blank_pattern = np.zeros((args.edge_length * 4, args.edge_length * 3, 3),np.uint8)
        if args.image_num == 1:
            begin_pos = [0, args.edge_length]
        elif args.image_num == 2:
            begin_pos = [args.edge_length * 3, args.edge_length]
        elif args.image_num == 3:
            begin_pos = [args.edge_length * 2, args.edge_length]
        elif args.image_num == 4:
            begin_pos = [args.edge_length, 0]
        elif args.image_num == 5:
            begin_pos = [args.edge_length, args.edge_length * 2]
        for i in range(64):
            now_pattern = blank_pattern.copy()
            now_pos = [begin_pos[0] + i % 8 * 16, begin_pos[1] + i // 8 * 16]
            now_pattern[now_pos[0]:now_pos[0] + 16, now_pos[1]:now_pos[1] + 16, :] = 255
            cv2.imwrite(args.pattern_root+"{}.png".format(i),now_pattern)
    
    if args.task == "calibrate_light":
        blank_pattern = np.zeros((args.edge_length * 4, args.edge_length * 3, 3),np.uint8)
        if args.image_num == 1:
            begin_pos = [0, args.edge_length]
        elif args.image_num == 2:
            begin_pos = [args.edge_length * 3, args.edge_length]
        elif args.image_num == 3:
            begin_pos = [args.edge_length * 2, args.edge_length]
        elif args.image_num == 4:
            begin_pos = [args.edge_length, 0]
        elif args.image_num == 5:
            begin_pos = [args.edge_length, args.edge_length * 2]
        for i in range(0, 1024):
            now_pattern = blank_pattern.copy()
            now_pos = [begin_pos[0] + i % 32 * 4, begin_pos[1] + i // 32 * 4]
            now_pattern[now_pos[0], now_pos[1], :] = 255
            cv2.imwrite(args.pattern_root+"{}.png".format(i - 0),now_pattern)
    
    if args.task == "test":
        blank_pattern = np.zeros((args.edge_length * 4, args.edge_length * 3, 3),np.uint8)
        if args.image_num == 1:
            begin_pos = [0, args.edge_length]
        elif args.image_num == 2:
            begin_pos = [args.edge_length * 3, args.edge_length]
        elif args.image_num == 3:
            begin_pos = [args.edge_length * 2, args.edge_length]
        elif args.image_num == 4:
            begin_pos = [args.edge_length, 0]
        elif args.image_num == 5:
            begin_pos = [args.edge_length, args.edge_length * 2]
        
        for k in range(3):
            for i in range(128):
                for j in range(128):
                    blank_pattern[i + begin_pos[0]][j + begin_pos[1]][k % 3] = int((127 - i) * j / 127 ** 2 * 255)
                    blank_pattern[i + begin_pos[0]][j + begin_pos[1]][(k + 1) % 3] = int(i * (127 - j) / 127 ** 2 * 255)
                    blank_pattern[i + begin_pos[0]][j + begin_pos[1]][(k + 2) %3] = int(max(i * j / 127 ** 2 * 255, (127 - i) * (127 - j) / 127 ** 2 * 255))
            cv2.imwrite(args.pattern_root+"{}.png".format(k),blank_pattern)               

    if args.task == "light_pos":
        pattern_counter = 0
        for which_row in range(args.edge_length * 4):
            for which_col in range(args.edge_length * 3):
                if (which_row < args.edge_length or which_row >= args.edge_length * 2) and (which_col < args.edge_length or which_col >= args.edge_length * 2):
                    continue
                else:
                    tmp_img = np.zeros((args.edge_length * 4, args.edge_length * 3, 3),np.uint8)
                    tmp_img[which_row, which_col, :] = 255 
                    cv2.imwrite(args.pattern_root+"{}.png".format(pattern_counter),tmp_img)
                    pattern_counter+=1