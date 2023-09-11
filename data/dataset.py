import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import time

import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
import load_frames_from_video as loadframe


class MY_DATASET(Dataset):
    def __init__(self, audio_dir, csv_file, n):
        self.csv = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.N = n

    def __getitem__(self, index):
        star = time.time()
        video_path = self.csv.values[index, 0]
        video_name = os.path.split(video_path)[1]
        label = self.csv.values[index, 1:]
        audio_path = os.path.join(self.audio_dir, video_name + ".wav_mt.csv")
        audio_feat = torch.tensor(pd.read_csv(audio_path, header = None).values[:, 0:68], dtype=torch.float32)
        # 数据归一化
        audio_feat = (audio_feat - audio_feat.min()) / (audio_feat.max() - audio_feat.min())

        # load frmaes and flows of video in video_path 
        video, total_frames = loadframe.open_video(video_path=video_path)
        sampler = loadframe.random_sample(self.N, total_frames=total_frames)
        frames_list = loadframe.get_img_from_index(index=sampler, video=video)
        # flow_star_time = time.time()
        # flows_list = loadframe.get_flow_from_frames(frames_list=frames_list)
        flow_end_time = time.time()
        # print("提取光流花费了{:.2f}秒".format(flow_end_time - flow_star_time))
        # cropped_face_rgb, cropped_face_flow = loadframe.get_image_face(frames_list=frames_list, flows_list=flows_list)
        cropped_face_rgb, cropped_face_flow = loadframe.get_image_face(frames_list=frames_list)
        face_end_time = time.time()
        # print("提取人脸花费了{:.2f}秒".format(face_end_time - flow_end_time))
        while(cropped_face_rgb.shape[0] < 6):
            sampler = loadframe.random_sample(self.N, total_frames=total_frames)
            frames_list = loadframe.get_img_from_index(index=sampler, video=video)
            # flows_list = loadframe.get_flow_from_frames(frames_list=frames_list)
            cropped_face_rgb, cropped_face_flow = loadframe.get_image_face(frames_list=frames_list)
        audio_output = audio_feat / audio_feat.mean()
        label = torch.tensor(label.tolist(), dtype=torch.float32)
        end = time.time()
        # print("提取一组数据共花费时间{:.2f}秒".format(end - star))

        return cropped_face_rgb, cropped_face_flow, audio_output, label   
    
    def __len__(self):
        return self.csv.values.shape[0]
        

