import argparse
import os
import yaml
import torch
import numpy as np
from mylogger import print_highlight, print_warning
from dataset import BaseKITTIDataset, KITTI_perturb
from utils.transform import UniformTransformSE3
from torch.utils.data import DataLoader
from models.CalibNet import CalibNet
from mylogger import get_logger
import loss as loss_utils


def options():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument("--dataset_path", type=str, default="data/")
    parser.add_argument("--skip_frame", type=int, default=5, help="skip frame of dataset")
    parser.add_argument("--pcd_sample", type=int, default=20000)
    parser.add_argument("--max_deg", type=float, default=10)  # 10deg in each axis (see the paper)
    parser.add_argument("--max_tran", type=float, default=0.2)  # 0.2m in each axis
    parser.add_argument("--max_randomly", type=bool, default=True)
    # dataloader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--pin_memory", type=bool, default=False,
                        help="set it to False if your CPU memory is insufficient")
    # schedule
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--log_dir", default="log/")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint/")
    parser.add_argument("--name", type=str, default="cam2_oneter_11to17")
    parser.add_argument("--optim", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-6)
    parser.add_argument("--lr_exp_decay", type=float, default=0.98)
    parser.add_argument("--clip_grad", type=float, default=2.0)
    # setting
    parser.add_argument("--scale", type=float, default=50.0,
                        help="scale factor of pcd normlization in loss")
    parser.add_argument("--inner_iter", type=int, default=1, help="inner iter of calibnet")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight of photo loss")
    parser.add_argument("--beta", type=float, default=0.3, help="weight of chamfer loss")
    parser.add_argument("--resize_ratio", type=float, nargs=2, default=[1.0, 1.0])
    # if CUDA is out of memory, please reduce batch_size, pcd_sample or inner_iter
    return parser.parse_args()


def train(args, chkpt, train_loader: DataLoader, val_loader: DataLoader):
    device = torch.device(args.device)
    model = CalibNet(backbone_pretrained=False, depth_scale=args.scale)
    model = model.to(device)
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr0, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_exp_decay)
    if args.pretrained:
        if os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
            model.load_state_dict(torch.load(args.pretrained)["model"])
            print_highlight("Pretrained model loaded from {}".format(args.pretrained))
        else:
            print_warning("Pretrained model not found at {}".format(args.pretrained))
    if chkpt is not None:
        model.load_state_dict(chkpt["model"])
        optimizer.load_state_dict(chkpt["optimizer"])
        scheduler.load_state_dict(chkpt["scheduler"])
        start_epoch = chkpt["epoch"] + 1
        min_loss = chkpt["min_loss"]
        log_mode = "a"
    else:
        start_epoch = 0
        min_loss = float("inf")
        log_mode = "w"
    if not torch.cuda.is_available():
        args.device = "cpu"
        print_warning("CUDA is not available, use CPU to run")
    log_mode = "a" if chkpt is not None else "w"
    logger = get_logger("{name}|Train".format(name=args.name), os.path.join(args.log_dir, args.name + ".log"), log_mode)
    if chkpt is None:
        logger.debug(args)
        print_highlight("Start training from epoch {}".format(start_epoch))
    else:
        print_highlight("Resume from epoch {}".format(start_epoch))
    del chkpt
    photo_loss = loss_utils.Photo_Loss(args.scale)
    alpha = float(args.alpha)
    beta = float(args.beta)
    for epoch in range(start_epoch, args.epoch):
        model.train()


if __name__ == "__main__":
    args = options()
    # 创建日志目录，如果该目录已经存在则不会抛出异常
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(args.config, "r") as f:
        # 使用 yaml 库读取配置文件内容，并使用 SafeLoader 加载，确保安全
        CONFIG = yaml.load(f, yaml.SafeLoader)
        # 确保配置文件内容是字典格式
    assert isinstance(CONFIG, dict), "Unknown config format!"
    if args.resume:
        # 加载指定的恢复检查点文件，通常是一个包含模型权重和配置的文件，将模型加载道 CPU 上，而不是 GPU
        chkpt = torch.load(args.resume, map_location="cpu")
        # 从恢复的检查点中获取配置并更新当前的 CONFIG 字典，确保训练恢复时使用的是之前的配置
        CONFIG.update(chkpt["config"])
        # 从恢复的检查点中获取参数并更新当前的args对象，args.__dict__ 访问的是 args 对象的字典属性，允许动态地更新其内容
        args.__dict__.update(chkpt["args"])
        print_highlight("config updated from resumed checkpoint {:s}".format(args.resume))
    else:
        chkpt = None
    print_highlight("args have been received, please wait for dataloader...")
    train_split = [str(index).rjust(2, "0") for index in CONFIG["dataset"]["train"]]
    val_split = [str(index).rjust(2, "0") for index in CONFIG["dataset"]["val"]]
    # dataset
    train_dataset = BaseKITTIDataset(args.dataset_path, args.batch_size, train_split, CONFIG["dataset"]["cam_id"],
                                     skip_frame=args.skip_frame, voxel_size=CONFIG["dataset"]["voxel_size"],
                                     pcd_sample_num=args.pcd_sample,
                                     resize_ratio=args.resize_ratio,
                                     extend_ratio=CONFIG["dataset"]["extend_ratio"]
                                     )
    train_dataset = KITTI_perturb(train_dataset, args.max_deg, args.max_tran, args.max_randomly,
                                  pooling_size=CONFIG["dataset"]["pooling"])

    val_dataset = BaseKITTIDataset(args.dataset_path, args.batch_size, val_split, CONFIG["dataset"]["cam_id"],
                                   skip_frame=args.skip_frame, voxel_size=CONFIG["dataset"]["voxel_size"],
                                   pcd_sample_num=args.pcd_sample, resize_ratio=args.resize_ratio,
                                   extend_ratio=CONFIG["dataset"]["extend_ratio"])
    val_perturb_file = os.path.join(args.checkpoint_dir, "val_seq.csv")
    val_length = len(val_dataset)
    if not os.path.exists(val_perturb_file):
        print_highlight("validation pertub file doesn't exist, create one.")
        transform = UniformTransformSE3(args.max_deg, args.max_tran, args.max_randomly)
        perturb_arr = np.zeros([val_length, 6])
        for i in range(val_length):
            perturb_arr[i, :] = transform.generate_transform().cpu().numpy()
        np.savetxt(val_perturb_file, perturb_arr, delimiter=",")
    else:  # check length
        val_seq = np.loadtxt(val_perturb_file, delimiter=",")
        if val_length != val_seq.shape[0]:
            print_warning("Incompatible validation length {} != {}".format(val_length, val_seq.shape[0]))
            transform = UniformTransformSE3(args.max_deg, args.max_tran, args.max_randomly)
            perturb_arr = np.zeros([val_length, 6])
            for i in range(val_length):
                perturb_arr[i, :] = transform.generate_transform().cpu().numpy()
            np.savetxt(val_perturb_file, perturb_arr, delimiter=",")
            print_highlight("validation perturb file rewritten.")

    val_dataset = KITTI_perturb(val_dataset, args.max_deg, args.max_tran, args.max_randomly,
                                pooling_size=CONFIG["dataset"]["pooling"],
                                file=os.path.join(args.checkpoint_dir, "val_seq.csv"))
    # batch normlization does not support batch = 1
    train_drop_last = True if len(train_dataset) % args.batch_size == 1 else False
    val_drop_last = True if len(val_dataset) % args.batch_size == 1 else False
    # dataloader
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                  pin_memory=args.pin_memory, drop_last=train_drop_last)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers + 8,
                                pin_memory=args.pin_memory, drop_last=val_drop_last)

    train(args, chkpt, train_dataloader, val_dataloader)