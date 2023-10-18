import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from load_img_threads import load_img


def _load_audio_input(audio_dir, video_name, suffix=".wav_st.csv"):
    audio_path = os.path.join(audio_dir, video_name + suffix)
    audio_input = pd.read_csv(audio_path, header = None).values
    audio_input = torch.as_tensor(audio_input, dtype=torch.float32)

    audio_tansforms = transforms.Normalize(mean=[0], std=[1])
    audio_input = audio_tansforms(audio_input)

    return audio_input

def _load_video_input(video_path, sampler, num_img, num_threads=4):
    frames_list = os.listdir(video_path)
    frames_list.sort()
    sampler = _get_sampler(total=len(frames_list), num_sam=num_img)
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
    def __init__(self, video_dir, flow_dir, audio_dir, csv_file, num_flow, n):
        self.csv = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.flow_dir = flow_dir
        self.num_flow = num_flow
        self.N = n

    def __getitem__(self, index):
        video_name = self.csv.values[index, 0]

        video_path = os.path.join(self.video_dir, video_name.replace(".mp4", ''))
        sampler = _get_sampler(path=video_path,
                               num_sam=self.N)

        label = self.csv.values[index, 1:]
        label = torch.as_tensor(label, dtype=torch.float32)

        audio_input = _load_audio_input(self.audio_dir,
                                        video_name=video_name)

        video_input = _load_video_input(video_path=video_path,
                                        sampler=sampler,
                                        num_img=self.N)

            
        label = label.tolist()

        return video_input, audio_input, label
    
    def __len__(self):
        return self.csv.values.shape[0]
        

