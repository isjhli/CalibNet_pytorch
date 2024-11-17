import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from utils import so3
from scipy.spatial.transform import Rotation
import numpy as np


class Photo_Loss(nn.Module):
    def __init__(self, scale=1.0, reduction='mean'):
        super(Photo_Loss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Unknown or invalid reduction"
        self.scale = scale
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Photo loss

        :param input: [B, H, W]
        :param target: [B, H, W]
        :return: scaled mse loss between input and target
        """
        return F.mse_loss(input / self.scale, target / self.scale, reduction=self.reduction)

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward(input, target)


class ChamferDistance(nn.Module):
    def __init__(self, scale=1.0, reduction='mean'):
        super(ChamferDistance, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Unknown or invalid reduction"
        self.scale = scale
        self.reduction = reduction

    def forward(self, template, source):
        p0 = template / self.scale
        p1 = source / self.scale
        if self.reduction == "none":
            return chamfer_distance(p0, p1)