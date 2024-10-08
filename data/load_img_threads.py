import threading
import os
from PIL import Image
import torch
from torchvision import transforms
import time
import numpy as np
import cv2

lock = threading.Lock()

transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize([224,224]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

class Load_imgs_thread(threading.Thread):
    def __init__(self):
        super().__init__()
    
    def run(self):
        global imgs_path
        global imgs
        with lock:
            while(imgs_path):  
                # img_path = imgs_path.pop()
                # with open(img_path, "rb") as f:
                #     data = f.read()
                # img_str = np.frombuffer(data, dtype=np.uint8)
                # img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
                # img = transform(img)
                # img = img.unsqueeze(dim=0)
                # if imgs is None:
                #     imgs = img
                # else:
                #     imgs = torch.cat((img, imgs), dim=0)

                img_path = imgs_path.pop()
                img = Image.open(img_path)
                img = transform(img)
                img = img.unsqueeze(dim=0)
                if imgs is None:
                    imgs = img
                else:
                    imgs = torch.cat((img, imgs), dim=0)

def load_img(frames_path:str, num_threads:int):
    """
    load imgs through multithreading
    """
    global imgs, imgs_path
    imgs = None
    imgs_path = frames_path
    t_list = []
    for i in range(num_threads):
        t = Load_imgs_thread()
        t_list.append(t)
        t.start()
    for t in t_list:
        t.join()

    return imgs
