# CalibNet_pytorch: Pytorch implementation of CalibNet

Learn from [https://github.com/gitouni/CalibNet_pytorch](https://github.com/gitouni/CalibNet_pytorch)

Original github: [https://github.com/epiception/CalibNet](https://github.com/epiception/CalibNet)

Original paper: [CalibNet: Geometrically Supervised Extrinsic Calibration using 3D Spatial Transformer Networks](https://arxiv.org/abs/1803.08181)



## Recommended Environment

Ubuntu  20.04

CUDA  11.1.1

cuDNN  8.9.0

PyTorch  1.8.0

Python  3.8

### Usage

Create the virtual environment:

`conda create -n <env_name> python=3.8`

`conda activate <env_name>`

Install Pytorch for CUDA 11.1:

`conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge`

Install dependencies

`pip install -r requirements.txt`

## Dataset Preparation

KITTI Odometry

Dataset should be organized into `data/`filefolder in root:

```
/PATH_TO_CalibNet_pytorch/
	--|data/
		--|poses/
			--|00.txt
			--|01.txt
			--...
		--|sequences/
			--|00/
				--|image_2/
				--|image_3/
				--|velodyne/
				--|calib.txt
				--|times.txt
			--|01/
			--|02/
			--...
	--...
```

Use [demo_pykitti.py](./demo_pykitti.py) to check your data. 



## Train and Test

### Train

The following command is fit with a 12GB GPU.

```bash
python train.py --batch_size=8 --epoch=100 --inner_iter=1 --pcd_sample=4096 --name=cam2_oneiter --skip_frame=10
```

### Test

```bash
python test.py --inner_iter=1 --pretrained=./checkpoint/cam2_oneiter_best.pth --skip_frame=1 --pcd_sample=-1
```



## Other Settings

see `config.yml` for dataset setting.

```yaml
dataset:
  train: [0,1,2,3,4,5,6,7]
  val: [8,9,10]
  test: [11,12,13]
  cam_id: 2  # (2 or 3)
  pooling: 3 # max pooling of semi-dense image, must be odd

```

- KITTI Odometry has 22 sequences, and you need to split them into three categories for training, validation and testing in `config.yml`
- `cam_id=2` represents left color image dataset and `cam_id=3` represents the right.
- set `pooling` paramter (only support odd numbers) to change max pooling of preprocessing for depth map.
