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
1. Download bounding-box detection models: [yolox-s](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw) and [yolox-m](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y), then make director `mono_tracking/scripts/AlphaPose/YOLOX/weights` and put the checkpoints to it.
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
@article{ye2023robot,
  title={Robot Person Following Under Partial Occlusion},
  author={Ye, Hanjing and Zhao, Jieting and Pan, Yaling and Chen, Weinan and He, Li and Zhang, Hong},
  journal={arXiv preprint arXiv:2302.02121},
  year={2023}
}
```

## Acknowledgement
- [monocular_person_following](https://github.com/koide3/monocular_person_following)
- [YOLOX_deepsort_tracker](https://github.com/pmj110119/YOLOX_deepsort_tracker)
- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
