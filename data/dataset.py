from einops import rearrange
from matplotlib.pylab import f
from data.load_img_threads import load_img
import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import time
from PIL import Image

import sys
sys.path.append(os.path.dirname(__file__))

lose_video = ['syTTeox8Yaw.003', 'xyWpSrfFlQw.004',
              'aaDlp62qn60.002', 'CFK8ib0aWe8.004']

transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Resize([224, 224]),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def normalization(input):
    """
    normalize the input to (-1,1)
    """

    max = input.max()
    min = input.min()
    output = ((input - min) / (max - min)) * 2 - 1

    return output


def trim_seq(input, l):
    """
    trim the sequence to the specified length
    """
    if input.shape[0] > l:
        input = input[0:l, :]
    else:
        len = l - input.shape[0]
        add = torch.zeros([len, input.shape[-1]])
        input = torch.cat((input, add), axis=0)
    return input


def _load_audio_input(audio_dir, video_name, use_6MCFF: bool):

    suffix = ".wav_mt.csv" if use_6MCFF else ".wav_st.csv"
    audio_path = os.path.join(audio_dir, video_name + suffix)
    audio_input = pd.read_csv(audio_path, header=None).values[:, :68]
    audio_input = torch.as_tensor(audio_input, dtype=torch.float32)

    audio_input = normalization(audio_input)
    if use_6MCFF:
        return audio_input
    audio_input = trim_seq(input=audio_input, l=300)

    return audio_input


def _load_video_input(video_path, frames_list, sampler, num_threads=2):
    frames_list.sort()
    frames_path = []
    for i in sampler:
        try:
            frames_path.append(os.path.join(video_path, frames_list[i]))
        except IndexError as err:
            print(f"video_name is {video_path}, the frame is {i} is None!!!")
    # for i in sampler:
    imgs = None
    for img_path in frames_path:
        img = Image.open(img_path)
        img = transform(img)
        img = img.unsqueeze(dim=0)
        if imgs is None:
            imgs = img
        else:
            imgs = torch.cat((imgs, img), dim=0)

    return imgs


def _get_sampler(total, num_sam):
    """
    Given an array range total and a quantity n, sample n portions within the number range total
    """
    par = total // num_sam
    sampel = torch.rand(num_sam)
    sampler = torch.zeros_like(sampel, dtype=torch.int)
    for index, num in enumerate(sampel):
        sam = (index * par + par * num).to(dtype=torch.int)
        sampler[index] = sam

    return sampler


def _load_video_stack(video_path, frames_list, sampler):
    """ According to the sampling strategy of sample, N image sets are read. Each image set is n consecutive images
    """
    stack = None
    samplers = [[(s + i) for i in range(5)] for s in sampler]
    for sam in samplers:
        imgs = _load_video_input(video_path=video_path,
                                 frames_list=frames_list, sampler=sam)
        imgs = rearrange(imgs, 'n c h w -> (n c) h w')
        imgs = imgs.unsqueeze(dim=0)
        if stack is None:
            stack = imgs
        else:
            stack = torch.cat((stack, imgs), dim=0)

    return stack


class MY_DATASET(Dataset):
    def __init__(self, cfg, is_train: bool):
        self.csv_path = cfg.train_csv_path if is_train else cfg.val_csv_path
        self.csv = pd.read_csv(self.csv_path)
        self.audio_dir = cfg.train_audio_dir if is_train else cfg.val_audio_dir
        self.video_dir = cfg.train_video_dir if is_train else cfg.val_video_dir
        self.global_dir = cfg.train_global_dir if is_train else cfg.val_global_dir
        self.flow_dir = cfg.train_flow_dir if is_train else cfg.val_flow_dir
        self.num_flow = cfg.num_flow
        self.N = cfg.N
        self.videos_name = self.csv["VideoName"]
        self.cfg = cfg

    def __getitem__(self, index):
        video_name = self.videos_name[index]

        if video_name.replace(".mp4", "") in lose_video:
            video_name = self.videos_name[index+1]

        # video_path = os.path.join(self.video_dir, video_name.replace(
        #     ".mp4", ''))
        video_path = os.path.join(self.video_dir, video_name.replace(
            ".mp4", ''), video_name.replace(".mp4", '') + "_aligned")

        imgs = [img for img in os.listdir(
            video_path) if img != '.ipynb_checkpoints']
        total = len(imgs) - 5

        # global_path = os.path.join(self.global_dir, video_name.replace(
        #     ".mp4", ''))
        # globals = [glo for glo in os.listdir(
        #     global_path) if glo != '.ipynb_checkpoints']

        sampler = _get_sampler(total=total, num_sam=self.N)
        if any(num < 0 for num in sampler):
            print(sampler, video_name)

        label = self.csv.values[index, 1:].astype(dtype=np.float32)
        label = torch.as_tensor(label, dtype=torch.float32)

        audio_input = _load_audio_input(audio_dir=self.audio_dir,
                                        video_name=video_name,
                                        use_6MCFF=self.cfg.use_6MCFF)

        video_input = _load_video_stack(video_path=video_path,
                                        frames_list=imgs,
                                        sampler=sampler,
                                        )

        # global_input = _load_video_input(video_path=global_path,
        #                                  frames_list=globals,
        #                                  sampler=sampler)

        # return video_input, global_input, audio_input, label

        return (video_input, audio_input), label

    def __len__(self):
        return self.csv.values.shape[0]
