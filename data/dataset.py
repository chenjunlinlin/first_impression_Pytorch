import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import time

import sys
sys.path.append(os.path.dirname(__file__))
from load_img_threads import load_img

lose_video = ['syTTeox8Yaw.003', 'xyWpSrfFlQw.004']

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

def _load_audio_input(audio_dir, video_name, suffix=".wav_st.csv"):
    audio_path = os.path.join(audio_dir, video_name + suffix)
    audio_input = pd.read_csv(audio_path, header = None).values
    audio_input = torch.as_tensor(audio_input, dtype=torch.float32)

    audio_input = normalization(audio_input)
    audio_input = trim_seq(input=audio_input, l=300)

    return audio_input

def _load_video_input(video_path, frames_list, sampler, num_threads=2):
    frames_list.sort()
    frames_path = []
    for i in sampler:
        frames_path.append(os.path.join(video_path, frames_list[i]))
    video_feat = load_img(frames_path=frames_path,
                          num_threads=num_threads).clone()

    return video_feat

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

class MY_DATASET(Dataset):
    def __init__(self, cfg, is_train:bool):
        self.csv_path = cfg.train_csv_path if is_train else cfg.val_csv_path
        self.csv = pd.read_csv(self.csv_path)
        self.audio_dir = cfg.train_audio_dir if is_train else cfg.val_audio_dir
        self.video_dir = cfg.train_video_dir if is_train else cfg.val_video_dir
        self.flow_dir = cfg.train_flow_dir if is_train else cfg.val_flow_dir
        self.num_flow = cfg.num_flow
        self.N = cfg.N
        self.videos_name = self.csv["VideoName"]
        self.cfg = cfg

    def __getitem__(self, index):
        video_name = self.videos_name[index]
        
        if video_name.replace(".mp4", "") in lose_video:
            video_name = self.videos_name[index+1]

        video_path = os.path.join(self.video_dir, video_name.replace(".mp4", ''))

        imgs = [img for img in os.listdir(video_path) if img != '.ipynb_checkpoints']
        total = len(imgs)

        sampler = _get_sampler(total=total,
                               num_sam=self.N)

        label = self.csv.values[index, 1:].astype(dtype=np.float32)
        label = torch.as_tensor(label, dtype=torch.float32)

        audio_input = _load_audio_input(audio_dir=self.audio_dir,
                                        video_name=video_name)

        video_input = _load_video_input(video_path=video_path,
                                        frames_list=imgs,
                                        sampler=sampler,
                                        num_threads=self.cfg.num_threads)

        return video_input, audio_input, label
    
    def __len__(self):
        return self.csv.values.shape[0]
        

