# Mono-RPF
[Robot Peron Following Under Partial Occlusion](https://sites.google.com/view/rpfpartial)

## Install
**Prequities**
- ROS, verified in melodic and noetic
- OpenCV with 3.4.12
- Ceres

1. Create a conda environment and install pytorch
```
conda create -n mono_following python=3.8
conda activate mono_following
# This is based on your GPU settings, other settings should be careful
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

2. Install python related packages:
```
pip install -r requirements.txt
git clone https://github.com/eric-wieser/ros_numpy
cd ros_numpy
python setup.py install
```

3. Install cpp related packages:
- OpenCV==3.4.12
- Eigen==3.0+

## Download pre-trained weights
1. Download bounding-box detection models from [Google Drive](https://drive.google.com/drive/folders/1a-z4zPpZX6XVwtklhTybydMcgH9JqZKR?usp=drive_link) or [YOLOX_deepsort_tracker], then make director `mono_tracking/scripts/AlphaPose/YOLOX/weights` and put the checkpoints to it.
2. Download 2d joint detection models: [Google drive](https://drive.google.com/drive/folders/1v-2Noym5U13BG6Zwj9EoqYRn6GXimh6p?usp=sharing), then make directory `mono_tracking/scripts/AlphaPose/Models` and put the checkpoints to it.

## How to use

Run with our self-built dataset as ROSBAG:
```bash
roslaunch mono_tracking all_mono_tracking.launch sim:=true
# play bag
rosbag play --clock -r 0.2 2022-07-15-17-09-34.bag
```

Run with our DINGO:
```bash
roslaunch mono_tracking all_mono_tracking.launch sim:=false
```


Run with icvs datasets as ROSBAG, and evaluate:
```bash
# If run in "corridor_corners" scene
roslaunch mono_tracking evaluate_MPF_in_icvs.launch scene:=corridor_corners
```
## Citation
```
@inproceedings{ye2023robot,
  title={Robot Person Following Under Partial Occlusion},
  author={Ye, Hanjing and Zhao, Jieting and Pan, Yaling and Chen, Weinan and He, Li and Zhang, Hong},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={7591--7597},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgement
- [monocular_person_following](https://github.com/koide3/monocular_person_following)
- [YOLOX_deepsort_tracker](https://github.com/pmj110119/YOLOX_deepsort_tracker)
- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
