import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
import load_frames_from_video as loadframe


class MY_DATASET(Dataset):
    def __init__(self, video_dir, audio_dir, csv_file, n):
        self.csv = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.N = n

    def __getitem__(self, index):
        video_path = self.csv.values[index, 0]
        video_name = os.path.split(video_path)
        label = self.csv.values[index, 1:]
        audio_path = os.path.join(self.audo_dir, video_name + ".wav_mt.csv")
        audio_feat = torch.tensor(pd.read_csv(audio_path, header = None).values[:,0:68])
        # 数据归一化
        audio_feat = (audio_feat - audio_feat.min()) / (audio_feat.max() - audio_feat.min())

        # load frmaes and flows of video in video_path 
        video, total_frames = loadframe.open_video(video_path=video_path)
        sampler = loadframe.random_sample(self.N, total_frames=total_frames)
        frames_list = loadframe.get_img_from_index(index=sampler, video=video)
        flows_list = loadframe.get_flow_from_frames(frames_list=frames_list)
        cropped_face_rgb, cropped_face_flow = loadframe.get_image_face(frames_list=frames_list, flows_list=flows_list)
        # 将光流信息填充为3维
        padding_data = torch.zeros((1, 1, cropped_face_flow.shape[2],
                                    cropped_face_flow.shape[3]))
        cropped_face_flow = torch.cat((cropped_face_flow, padding_data), dim=1)
        audio_output = torch.tensor(np.array(audio_feat), 
                                    dtype=torch.float32)
        audio_output = audio_output / audio_output.mean()
        label = torch.tensor(label.tolist(), dtype=torch.float32)

        return cropped_face_rgb, cropped_face_flow, \
                audio_output, label
    
    def __len__(self):
        return self.csv.values.shape[0]
        

