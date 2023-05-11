import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.sh import mkdir, sh

BLOCK_SIZE = 64


def convert_patterns(pattern_dir: str, pattern_num: int, tmp_location: str):
    mkdir(tmp_location)
    for i in range(5):  # 5 faces
        mkdir(rf'{tmp_location}/face{i + 1}')

    for i in range(pattern_num):
        pt_img = cv2.imread(f'{pattern_dir}/{i}.png')
        print(pt_img.shape)
        converted_img = np.zeros((3 * BLOCK_SIZE, 4 * BLOCK_SIZE, 3))

        # Face 1
        big_block_slice = pt_img[BLOCK_SIZE * 0:BLOCK_SIZE * 2, BLOCK_SIZE * 2: BLOCK_SIZE * 4]
        # 1-1
        pt_slice = big_block_slice[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 0:BLOCK_SIZE * 1]
        pt_slice = np.swapaxes(pt_slice, 0, 1)[::-1, ::-1]
        converted_img[BLOCK_SIZE:BLOCK_SIZE * 2, BLOCK_SIZE: BLOCK_SIZE * 2] = pt_slice
        # 1-2
        pt_slice = big_block_slice[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 1:BLOCK_SIZE * 2]
        pt_slice = np.swapaxes(pt_slice, 0, 1)
        converted_img[BLOCK_SIZE:BLOCK_SIZE * 2, BLOCK_SIZE * 2: BLOCK_SIZE * 3] = pt_slice
        # 1-3
        pt_slice = big_block_slice[BLOCK_SIZE * 2 - 1:BLOCK_SIZE * 1 - 1:-1, BLOCK_SIZE * 0:BLOCK_SIZE * 1]
        converted_img[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 1: BLOCK_SIZE * 2] = pt_slice
        # 1-4
        pt_slice = big_block_slice[BLOCK_SIZE * 1:BLOCK_SIZE * 2, BLOCK_SIZE * 1:BLOCK_SIZE * 2]
        pt_slice = np.swapaxes(pt_slice, 0, 1)[::-1, ::-1]
        converted_img[BLOCK_SIZE * 1:BLOCK_SIZE * 2, BLOCK_SIZE * 0: BLOCK_SIZE * 1] = pt_slice
        cv2.imwrite(f'{tmp_location}/face1/W_{i}.png', converted_img)

        # Face 2
        big_block_slice = pt_img[BLOCK_SIZE * 6:BLOCK_SIZE * 8, BLOCK_SIZE * 2: BLOCK_SIZE * 4]
        # 2-1
        pt_slice = big_block_slice[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 0:BLOCK_SIZE * 1]
        pt_slice = np.swapaxes(pt_slice, 0, 1)  # [::-1, ::-1]
        converted_img[BLOCK_SIZE:BLOCK_SIZE * 2, BLOCK_SIZE: BLOCK_SIZE * 2] = pt_slice
        # 2-2
        pt_slice = big_block_slice[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 1:BLOCK_SIZE * 2]
        pt_slice = np.swapaxes(pt_slice, 0, 1)[::-1, ::-1]
        converted_img[BLOCK_SIZE:BLOCK_SIZE * 2, BLOCK_SIZE * 2: BLOCK_SIZE * 3] = pt_slice
        # 2-3
        pt_slice = big_block_slice[BLOCK_SIZE * 2 - 1:BLOCK_SIZE * 1 - 1:-1, BLOCK_SIZE * 0:BLOCK_SIZE * 1]
        pt_slice = pt_slice[::-1, ::-1]
        converted_img[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 1: BLOCK_SIZE * 2] = pt_slice
        # 2-4
        pt_slice = big_block_slice[BLOCK_SIZE * 1:BLOCK_SIZE * 2, BLOCK_SIZE * 1:BLOCK_SIZE * 2]
        pt_slice = np.swapaxes(pt_slice, 0, 1)  # [::-1, ::-1]
        converted_img[BLOCK_SIZE * 1:BLOCK_SIZE * 2, BLOCK_SIZE * 0: BLOCK_SIZE * 1] = pt_slice
        cv2.imwrite(f'{tmp_location}/face2/W_{i}.png', converted_img)

        # Face 3
        big_block_slice = pt_img[BLOCK_SIZE * 4:BLOCK_SIZE * 6, BLOCK_SIZE * 2: BLOCK_SIZE * 4]
        big_block_slice = big_block_slice[::-1, ::-1]
        # 3-1
        pt_slice = big_block_slice[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 0:BLOCK_SIZE * 1]
        pt_slice = np.swapaxes(pt_slice, 0, 1)[::-1, ::-1]
        converted_img[BLOCK_SIZE:BLOCK_SIZE * 2, BLOCK_SIZE: BLOCK_SIZE * 2] = pt_slice
        # 3-2
        pt_slice = big_block_slice[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 1:BLOCK_SIZE * 2]
        pt_slice = np.swapaxes(pt_slice, 0, 1)
        converted_img[BLOCK_SIZE:BLOCK_SIZE * 2, BLOCK_SIZE * 2: BLOCK_SIZE * 3] = pt_slice
        # 3-3
        pt_slice = big_block_slice[BLOCK_SIZE * 2 - 1:BLOCK_SIZE * 1 - 1:-1, BLOCK_SIZE * 0:BLOCK_SIZE * 1]
        converted_img[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 1: BLOCK_SIZE * 2] = pt_slice
        # 3-4
        pt_slice = big_block_slice[BLOCK_SIZE * 1:BLOCK_SIZE * 2, BLOCK_SIZE * 1:BLOCK_SIZE * 2]
        pt_slice = np.swapaxes(pt_slice, 0, 1)[::-1, ::-1]
        converted_img[BLOCK_SIZE * 1:BLOCK_SIZE * 2, BLOCK_SIZE * 0: BLOCK_SIZE * 1] = pt_slice
        cv2.imwrite(f'{tmp_location}/face3/W_{i}.png', converted_img)

        # Face 4
        big_block_slice = pt_img[BLOCK_SIZE * 2:BLOCK_SIZE * 4, BLOCK_SIZE * 0: BLOCK_SIZE * 2]
        big_block_slice = big_block_slice[::-1]
        big_block_slice = np.swapaxes(big_block_slice, 0, 1)
        # 4-1
        pt_slice = big_block_slice[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 0:BLOCK_SIZE * 1]
        pt_slice = np.swapaxes(pt_slice, 0, 1)[::-1, ::-1]
        converted_img[BLOCK_SIZE:BLOCK_SIZE * 2, BLOCK_SIZE: BLOCK_SIZE * 2] = pt_slice
        # 4-2
        pt_slice = big_block_slice[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 1:BLOCK_SIZE * 2]
        pt_slice = np.swapaxes(pt_slice, 0, 1)
        converted_img[BLOCK_SIZE:BLOCK_SIZE * 2, BLOCK_SIZE * 2: BLOCK_SIZE * 3] = pt_slice
        # 4-3
        pt_slice = big_block_slice[BLOCK_SIZE * 2 - 1:BLOCK_SIZE * 1 - 1:-1, BLOCK_SIZE * 0:BLOCK_SIZE * 1]
        converted_img[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 1: BLOCK_SIZE * 2] = pt_slice
        # 4-4
        pt_slice = big_block_slice[BLOCK_SIZE * 1:BLOCK_SIZE * 2, BLOCK_SIZE * 1:BLOCK_SIZE * 2]
        pt_slice = np.swapaxes(pt_slice, 0, 1)[::-1, ::-1]
        converted_img[BLOCK_SIZE * 1:BLOCK_SIZE * 2, BLOCK_SIZE * 0: BLOCK_SIZE * 1] = pt_slice
        cv2.imwrite(f'{tmp_location}/face4/W_{i}.png', converted_img)

        # Face 5
        big_block_slice = pt_img[BLOCK_SIZE * 2:BLOCK_SIZE * 4, BLOCK_SIZE * 4: BLOCK_SIZE * 6]
        big_block_slice = big_block_slice[:, ::-1]
        big_block_slice = np.swapaxes(big_block_slice, 0, 1)
        # 5-1
        pt_slice = big_block_slice[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 0:BLOCK_SIZE * 1]
        pt_slice = np.swapaxes(pt_slice, 0, 1)[::-1, ::-1]
        converted_img[BLOCK_SIZE:BLOCK_SIZE * 2, BLOCK_SIZE: BLOCK_SIZE * 2] = pt_slice
        # 5-2
        pt_slice = big_block_slice[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 1:BLOCK_SIZE * 2]
        pt_slice = np.swapaxes(pt_slice, 0, 1)
        converted_img[BLOCK_SIZE:BLOCK_SIZE * 2, BLOCK_SIZE * 2: BLOCK_SIZE * 3] = pt_slice
        # 5-3
        pt_slice = big_block_slice[BLOCK_SIZE * 2 - 1:BLOCK_SIZE * 1 - 1:-1, BLOCK_SIZE * 0:BLOCK_SIZE * 1]
        converted_img[BLOCK_SIZE * 0:BLOCK_SIZE * 1, BLOCK_SIZE * 1: BLOCK_SIZE * 2] = pt_slice
        # 5-4
        pt_slice = big_block_slice[BLOCK_SIZE * 1:BLOCK_SIZE * 2, BLOCK_SIZE * 1:BLOCK_SIZE * 2]
        pt_slice = np.swapaxes(pt_slice, 0, 1)[::-1, ::-1]
        converted_img[BLOCK_SIZE * 1:BLOCK_SIZE * 2, BLOCK_SIZE * 0: BLOCK_SIZE * 1] = pt_slice
        cv2.imwrite(f'{tmp_location}/face5/W_{i}.png', converted_img)

    for i in range(5):
        sh(rf'.\bin\pattern_converter.exe -p {tmp_location}\face{i + 1}\ -n {pattern_num}')


if __name__ == '__main__':
    convert_patterns(r'../demo_patterns', 1, r'../bin/tmp_pattern')
