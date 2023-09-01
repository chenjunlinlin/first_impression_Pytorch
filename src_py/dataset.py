import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MY_DATASET(Dataset):
    def __init__(self, video_dir, audio_dir, csv_file, n):
        self.csv = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.N = n

    def __getitem__(self, index):
        video_name = self.csv.values[index, 0]
        label = self.csv.values[index, 1:]
        audio_path = os.path.join(self.audio_dir, video_name + ".wav_mt.csv")
        video_path = os.path.join(self.video_dir, video_name.replace(".mp4", ''))
        audio_feat = torch.tensor(pd.read_csv(audio_path, header = None).values[:,0:68])
        # 数据归一化
        audio_feat = (audio_feat - audio_feat.min()) / (audio_feat.max() - audio_feat.min())
        video_frames_dir = os.path.join(video_path, video_name.replace(".mp4", '') + '_aligned')
        frames_list = os.listdir(video_frames_dir)
        frames_list.sort()
        partion_len = len(frames_list) // self.N
        video_feat = []
        for partition in range(self.N):
            num = torch.randint(partition * partion_len, (partition+1) * partion_len, ())
            if frames_list[num] == ".ipynb_checkpoints":
                num = torch.randint(partition * partion_len, (partition+1) * partion_len, ())
            img_path = os.path.join(video_frames_dir, frames_list[num])
            if os.path.exists(img_path):
                img = cv2.imread(img_path).transpose(2, 0, 1)
            else:
                print("图片读取失败：{}".format(img_path))
            video_feat.append(img)
        video_output = torch.as_tensor(np.array(video_feat)) / 255
        audio_output = torch.as_tensor(np.array(audio_feat))
        audio_output = audio_output / audio_output.mean()
        label = label.tolist()

        return video_output.float().transpose(0,1), audio_output.float(), torch.as_tensor(label).float()
    
    def __len__(self):
        return self.csv.values.shape[0]
        

