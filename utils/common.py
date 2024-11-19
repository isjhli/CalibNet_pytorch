import torch
import numpy as np
from matplotlib import pyplot as plt


def SaveProjImg(saveDir, rgb_img: torch.Tensor, pcd: torch.Tensor, pcd_range, InTran: torch.Tensor):
    """ Save the image with projected pcd to disk, only for batch_size=1.

    :param saveDir: the directory to save the image.
    :param rgb_img: the RGB image, (B, 3, H, W) where B should be equal to 1.
    :param pcd: the point cloud data, (B, 3, N) where N should be equal to 1.
    :param pcd_range: the point cloud distance, (B, N) where B should be equal to 1.
    :param InTran: the camera intrinsic matrix, (3, 3)
    """
    H, W = rgb_img.shape[-2:]
    img_draw = rgb_img.cpu().detach().numpy()[0, ...].transpose((1, 2, 0))
    proj_pcd = InTran.matmul(pcd[0, ...]).cpu().detach().numpy()  # [3, 3] x [3, N] -> [3, N]
    proj_x = (proj_pcd[0, :] / proj_pcd[2, :]).astype(np.int64)
    proj_y = (proj_pcd[1, :] / proj_pcd[2, :]).astype(np.int64)
    rev = (proj_x >= 0) * (proj_x < W) * (proj_y >= 0) * (proj_y < H) * (proj_pcd[2, :] > 0)
    proj_x = proj_x[rev]
    proj_y = proj_y[rev]
    proj_r = pcd_range[0, rev].cpu().detach().numpy()

    plt.figure(figsize=(12, 5), dpi=100, tight_layout=True)
    plt.cla()
    plt.axis([0, W, H, 0])
    plt.imshow(img_draw)
    plt.scatter([proj_x], [proj_y], c=[proj_r], cmap='rainbow_r',
                alpha=0.5, s=2)
    plt.savefig(saveDir, bbox_inches='tight')
    plt.close()
