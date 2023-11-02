from mimetypes import init
import torch
import os
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader as dataloader
import random
import numpy as np

from data import dataset


def get_opt(cfg, model, opt_para):
    """
    get a optimizer
    """
    if cfg.opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(
        ), weight_decay=cfg.weight_decay, lr=cfg.lr, momentum=cfg.momentum)
    if cfg.opt == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=cfg.weight_decay, lr=cfg.lr)
    if opt_para is not None:
        optimizer.load_state_dict(opt_para)

    return optimizer


def get_scheduler(cfg, opt, epoch):
    """
    get a scheduler
    """
    if epoch is not None:
        if cfg.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=opt, step_size=30, gamma=0.9, last_epoch=epoch)
        if cfg.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=opt, T_0=5, T_mult=2, eta_min=0.001, last_epoch=epoch)
    else:
        if cfg.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=opt, step_size=30, gamma=0.9)
        if cfg.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=opt, T_0=5, T_mult=2, eta_min=0.001)

    return scheduler


def get_model(cfg, model, exp_name):
    """
    load checkpoint for model
    """
    print('--------模型初始化---------')
    flag, checkpoint = get_checkpoint(cfg=cfg, exp_name=exp_name)
    if cfg._continue and flag:
        model = load_model(model=model, dict=checkpoint)
        print("模型加载成功，继续训练！！！\tEpoch:{}"
              .format(checkpoint["epoch"]))
    elif cfg.model_path is not None:
        dict = torch.load(cfg.model_path)
        model = load_model(model=model, dict=dict)
        print("模型加载成功，开始训练！！！\nmodel from:{}"
              .format(cfg.model_path))
    else:
        # init_model.initialize_weights(model)
        print(f"开始训练,Epoch:1")
    if cfg.iscuda:
        print("load model to cuda")
    if cfg.DP:
        model = nn.DataParallel(model.cuda(), device_ids=cfg.devices)
    else:
        model = model.cuda()

    if flag and cfg._continue:
        return model, checkpoint['epoch'], checkpoint["optimizer"]

    return model, None, None


def get_cudainfo():
    available_cuda_devices = [torch.device(
        f'cuda:{i}') for i in range(torch.cuda.device_count())]
    for device in available_cuda_devices:
        print(f"可用的 CUDA 设备：{device}")


def get_checkpoint(cfg, exp_name):
    """
    get checkpoint dict
    """
    checkpoints_path = os.path.join(cfg.model_save_dir, exp_name)
    if os.path.exists(checkpoints_path):
        checkpoints_list = [int(i.replace(".pth", ""))
                            for i in os.listdir(checkpoints_path)]
        checkpoints_list.sort()
        if len(checkpoints_list) == 0:
            return False, None
        num = checkpoints_list[-1]
        checkpoint_path = os.path.join(
            checkpoints_path, "{:04d}.pth".format(num))
        checkpoint = torch.load(checkpoint_path)

        return True, checkpoint

    return False, None


def load_model(model, dict):
    state_dict = dict['net']
    new_state_dict = OrderedDict()
    for k in state_dict:
        new_k = k.replace('module.', '')
        new_state_dict[new_k] = state_dict[k]
    model.load_state_dict(state_dict=new_state_dict)

    return model


def get_dataloader(cfg, is_train: bool):

    data_set = dataset.MY_DATASET(cfg=cfg, is_train=is_train)

    data_loader = dataloader(data_set, batch_size=cfg.batch_size,
                             shuffle=True, num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
                             worker_init_fn=np.random.seed())
    return data_loader


def update_checkpoint(checkpoint, net_para, opt_para, epoch):
    """
    update checkpoint info
    """
    checkpoint['net'] = net_para
    checkpoint['epoch'] = epoch
    checkpoint['optimizer'] = opt_para

    return checkpoint


def set_random_seed(seed=10, deterministic=True, benchmark=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
