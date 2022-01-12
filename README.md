# Dexi+RAFT
This file contains the source code for the research project:

Motion and Edge fusion via dense extrem inception network

##
This work combine a optical flow network 'RAFT' and an edge detection network 'DexiNed'.
* [RAFT] https://github.com/princeton-vl/RAFT
* [DexiNed] https://github.com/xavysp/DexiNed

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```Shell
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```
Python 3.7
Pytorch >=1.4 (Last test 1.9)
OpenCV
Matplotlib
Kornia
Other package like Numpy, h5py, PIL, json.

## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [BIPED] (https://xavysp.github.io/MBIPED/)


## Training
following training schedule in 'train_mixed.txt'. Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell

## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```
