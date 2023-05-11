from src.sh import sh


def flash(tmp_pattern_dir: str, begin_pattern:int,pattern_num: int, flash_duration: float,cam_cap_delay: float, save_root: str):
    sh(rf".\bin\little_star.exe --task 0 --t {flash_duration} --light --flashd -1.0 "
       rf"--camfpga 0 --patternnum {pattern_num} --beginpatternnum {begin_pattern} "
       rf"-p {tmp_pattern_dir}\ --pb tentacle_ -s {save_root} --ldr --camdelay {cam_cap_delay}")
