from dataclasses import dataclass
from simple_parsing import ArgumentParser
from src.sh import sh
import glob
import cv2
project_root = r"D:\project\ipad\data\calibrate_camera\20230506-20-53-20_jpeg/"
calibration_result_root = project_root + r"calibration_result/"
obj_name = r"frames/"

@dataclass
class Options:
    chessboard_corner_num: str = "7,11" #9*12 chessboard
    #chessboard_block_edge_size: float = 9.813636363636363
    chessboard_block_edge_size: float = 20.3
    save_root: str = project_root

def calibrate_intrinsic_cal(data_root: str, calibration_result_root: str, chessboard_corner_num: str, chessboard_block_edge_size: int, task: str):
    sh(rf"python .\src\calibrate_intrinsic.py --data_root {data_root} --calibration_result_root {calibration_result_root} "
       rf"--chessboard_corner_num {chessboard_corner_num} --chessboard_block_edge_size {chessboard_block_edge_size} "
       rf"--task {task}")
    
def calibrate_intrinsic(data_root: str, chessboard_corner_num: str, chessboard_block_edge_size: int, task: str):
    sh(rf"python .\src\calibrate_intrinsic.py --data_root {data_root} "
       rf"--chessboard_corner_num {chessboard_corner_num} --chessboard_block_edge_size {chessboard_block_edge_size} "
       rf"--task {task}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args()
    options = args.options
    #print("options:", options)

    imgs=glob.glob(f"{options.save_root}*.png")
    imgs=imgs+glob.glob(f"{options.save_root}*.jpeg")
    print(len(imgs))

    # #intrinsic
    # calibrate_intrinsic(
    #     data_root = options.save_root,
    #     chessboard_corner_num = options.chessboard_corner_num,
    #     chessboard_block_edge_size = options.chessboard_block_edge_size,
    #     task = "add_intrinsic_images"   
    # )

    # #extrinsic
    # calibrate_intrinsic(
    #     data_root = options.save_root,
    #     chessboard_corner_num = options.chessboard_corner_num,
    #     chessboard_block_edge_size = options.chessboard_block_edge_size,
    #     task = "add_extrinsic_image"
    # )

    #calibrate
    calibrate_intrinsic_cal(
        data_root = options.save_root,
        calibration_result_root = calibration_result_root,
        chessboard_corner_num = options.chessboard_corner_num,
        chessboard_block_edge_size = options.chessboard_block_edge_size,
        task = "calibrate"
    )