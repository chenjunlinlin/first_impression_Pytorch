import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms



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
        label = self.csv.values[index, 1:]
        audio_path = os.path.join(self.audio_dir, video_name + ".wav_mt.csv")
        audio_path = os.path.abspath(audio_path)
        video_path = os.path.join(self.video_dir, video_name.replace(".mp4", ''))
        flows_path = os.path.join(self.flow_dir, video_name.replace(".mp4", ''))
        video_path = os.path.abspath(video_path)
        audio_feat = torch.tensor(pd.read_csv(audio_path, header = None).values[:, :68], dtype=torch.float32)
        # 数据归一化
        audio_feat = (audio_feat - audio_feat.min()) / (audio_feat.max() - audio_feat.min())
        video_frames_dir = os.path.join(video_path, video_name.replace(".mp4", '') + '_aligned')
        frames_list = os.listdir(video_frames_dir)
        frames_list.sort()
        flows_list = os.listdir(flows_path)
        flows_list.sort()
        partion_len = len(frames_list) // self.N
        flow_partion_len = len(flows_list) // self.N
        video_feat = torch.zeros((self.N, 3, 224, 224))
        flow_feat = torch.zeros((self.N, 3, 224, 224))
        
        data_transfomr = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize([224,224]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        flow_transfomr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224,224]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        for partition in range(self.N):
            num = torch.randint(partition * partion_len, (partition+1) * partion_len, ())
            if frames_list[num] == ".ipynb_checkpoints":
                num = torch.randint(partition * partion_len, (partition+1) * partion_len, ())
            img_path = os.path.join(video_frames_dir, frames_list[num])
            if os.path.exists(img_path):
                try:
                    img = cv2.imread(img_path)
                    video_feat[partition] = data_transfomr(img)
                except AttributeError as error:
                    print('发生错误了：{}'.format(img_path))
            else:
                print("图片读取失败：{}".format(img_path))
                video_feat[partition] = torch.zeros((3, 224, 224))

            # # load flow frame
            # index_flow = torch.randint(partition * flow_partion_len, (partition+1) * flow_partion_len, ())
            # if flows_list[index_flow] == ".ipynb_checkpoints":
            #     index_flow = torch.randint(partition * flow_partion_len, (partition+1) * flow_partion_len, ())
            # if index_flow + self.num_flow > len(flows_list):
            #     index_flow -= self.num_flow
            # for i in range(index_flow, index_flow+self.num_flow):
            #     flow_path = os.path.join(flows_path, flows_list[i])
            #     flow = cv2.imread(flow_path)
            #     flow = flow_transfomr(flow)
            #     flow_feat[partition] += flow

            
        label = label.tolist()

        # return video_feat, flow_feat, audio_feat, torch.as_tensor(label, dtype=torch.float32)
        return video_feat, audio_feat, torch.as_tensor(label, dtype=torch.float32)
    
    def __len__(self):
        return self.csv.values.shape[0]
        

