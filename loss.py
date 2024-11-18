from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import numpy as np

from utils import so3
from losses.chamfer_loss import chamfer_distance


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


class ChamferDistanceLoss(nn.Module):
    def __init__(self, scale=1.0, reduction='mean'):
        super(ChamferDistanceLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Unknown or invalid reduction"
        self.scale = scale
        self.reduction = reduction

    def forward(self, template, source):
        p0 = template / self.scale
        p1 = source / self.scale
        if self.reduction == "none":
            return chamfer_distance(p0, p1)
        elif self.reduction == "mean":
            return torch.mean(chamfer_distance(p0, p1), dim=0)
        elif self.reduction == "sum":
            return torch.sum(chamfer_distance(p0, p1), dim=0)

    def __call__(self, template, source):
        return self.forward(template, source)


def geodesic_distance(x: torch.Tensor) -> tuple:
    """geodesic distance for evaluation

    :param x: [B, 4, ]
    :return: distance of component R and T
    """
    R = x[:, :3, :3]  # [B, 3, 3] rotation
    T = x[:, :3, 3]  # [B, 3] translation
    dR = so3.log(R)  # [B, 3]
    dR = F.mse_loss(dR, torch.zeros_like(dR).to(dR), reduction="none").mean(dim=0)  # [B, 3] -> [B, 1]
    dR = torch.sqrt(dR).mean(dim=0)  # [B, 1] -> [1,] Rotation RMSE (mean in batch)
    dT = F.mse_loss(T, torch.zeros_like(T).to(T), reduction="none").mean(dim=0)  # [B, 3] -> [B, 1]
    dT = torch.sqrt(dT).mean(dim=0)  # [B, 1] -> [1,] Translation RMSE (mean in batch)
    return dR, dT


def gt2euler(gt: np.ndarray):
    """gt transformer to euler angles and translation

    :param gt: [4, 4]
    :return: angle_gt [3, 1], trans_gt [3, 1]
    """
    R_gt = gt[:3, :3]
    euler_angle = Rotation.from_matrix(R_gt)
    anglez_gt, angley_gt, anglex_gt = euler_angle.as_euler("zyx")
    angle_gt = np.array([anglex_gt, angley_gt, anglez_gt])
    trans_gt_t = -R_gt @ gt[:3, 3]
    return angle_gt, trans_gt_t
