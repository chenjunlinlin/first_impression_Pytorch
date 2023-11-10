import pandas as pd
import os
import torch
import cv2
import numpy as np  
import dlib
from torchvision import transforms

def open_video(video_path):
    '''
    input:
        video_path
    return:
        video,total_frames
    '''
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("打开视频{}失败！".format(video_path))
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return video, total_frames

def random_sample(N,total_frames):
    '''
    input:
        N: N is the number of segments the video is divided into.
        total_frames: 
    return:
        index:is a list. Random indices within each portion.
    '''
    index = torch.rand(N)
    per = total_frames // N
    for i in range(N):
        index[i] = index[i]*per + i*per
    index = index.to(dtype=torch.int)
    return index

def get_img_from_index(index, video):
    '''
    input:
        index：is a list. Random indices within each portion
        video：return from cv2
    return:
        frames_list
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

def get_flow_from_frames(frames_list):
    '''
    input:
        frames_list
    return:
        flows_list
    '''
    flows_list = np.zeros((len(frames_list), frames_list[0][0].shape[0], frames_list[0][0].shape[1], 2), dtype=np.uint8)
    for i, frames in enumerate(frames_list):
        cur = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        pre = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        flow = compute_TVL1(pre, cur)
        flows_list[i] = flow
    return flows_list

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def get_image_face(frames_list, flows_list=None):
    '''
    get face region of flow and rgb
    cropped_face_rgb:(N, W, H, 3)
    cropped_face_flow:(N, 2, W, H)
    '''
    cropped_face_rgb = []*6
    cropped_face_flow = []*6
    face_detector = dlib.get_frontal_face_detector()
    height, width = frames_list.shape[2:4]
    frame_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
    flow_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224]),
        ])
    for i, frame in enumerate(frames_list):
        gray = cv2.cvtColor(frame[0], cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement

            x, y, size = get_boundingbox(face, width, height)
            cropped_face_rgb.append(frame_transform
                                    (frame[0][y:y+size, x:x+size, :]))
            if flows_list :
                flow = flow_transform(flows_list[i][y:y+size, x:x+size, :])
                # Fill the optical flow information into 3 dimensions
                padding_data = torch.zeros((1, 224, 224))
                flow = torch.cat((flow, padding_data))
                cropped_face_flow.append(flow)
    return torch.tensor(np.array(cropped_face_rgb), dtype=torch.float32),torch.tensor(np.array(cropped_face_flow), dtype=torch.float32)