import PIL.Image
import torch
from torchvision import datasets, transforms

from dataset import BaseKITTIDataset

dataset = BaseKITTIDataset("./data", 1, cam_id=0)

print(len(dataset))
img = dataset[0]

ii = transforms.ToPILImage()(img)
ii.show()
