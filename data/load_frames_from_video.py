import pandas as pd
import os
import torch
import cv2
import numpy as np  

def open_video(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("打开视频{}失败！".format(video_path))
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return video, total_frames

def random_sample(N,total_frames):
    index = torch.rand(N)
    per = total_frames // N
    print(index,per)
    for i in range(N):
        index[i] = index[i]*per + i*per
    index = index.to(dtype=torch.int)
    return index

def get_img_from_index(index, video):
    '''
        index：每份视频中帧的抽样
        video：cv2读取视频的返回值
    '''
    _, frame = video.read()
    frames_list = np.zeros((len(index), 2, frame.shape[0], frame.shape[1], frame.shape[2]), dtype=np.uint8)
    for i, ind in enumerate(index):
        if ind == 0:
            ind =1
        video.set(cv2.CAP_PROP_POS_FRAMES, float(ind))
        _, cur = video.read()
        frames_list[i, 0] = cur
        video.set(cv2.CAP_PROP_POS_FRAMES, float(ind-1))
        _, pre = video.read()
        frames_list[i,1] = pre
    return frames_list

def compute_TVL1(prev, curr, bound=15):
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)

    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow

def get_flow_from(frames_list):
    flows_list = np.zeros((len(frames_list), flows_list[0].shape[0], flows_list[0].shape[0], 2), dtype=np.uint8)
    for i, frames in enumerate(frames_list):
        cur = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        pre = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        flow = compute_TVL1(pre, cur)
        flows_list[i] = flow
    return flows_list