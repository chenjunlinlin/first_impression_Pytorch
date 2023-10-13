from torchvision.io import write_jpeg
import torch
from torchvision.io import read_video
import argparse
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large
import os
from torchvision.utils import flow_to_image
import warnings
warnings.filterwarnings("ignore")
import numpy as np

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if os.path.isdir(path):
            pass
        else:
            raise

def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch, img2_batch)


def process_video(video_path, video_name, destpath_flow ):
    frames, _, _ = read_video(video_path)
    frames = frames.permute(0, 3, 1, 2)
    video_name = video_name.replace(".mp4", "")

    # If you can, run this example on a GPU, it will be a lot faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
    # Note: it would be faster to predict batches of flows instead of individual flows
        img1, img2 = preprocess(img1, img2)
        img1, img2 = torch.unsqueeze(img1, dim=0), torch.unsqueeze(img2, dim=0)
        list_of_flows = model(img1.to(device), img2.to(device))
        predicted_flow = list_of_flows[-1][0]
        # print(predicted_flow.shape)
        # flow_img = predicted_flow.to("cpu").detach().squeeze(dim=0)
        # print(flow_img.shape)
        flow_img = flow_to_image(predicted_flow).to("cpu")
        # flow_img = flow_img.numpy()
        output_folder = os.path.join(destpath_flow, video_name)  # Update this to the folder of your choice
        mkdir_p(output_folder)
        write_jpeg(flow_img, os.path.join(output_folder , f"{i:04d}.jpg"))
        # np.save(os.path.join(output_folder , f"{i:04d}.npy"), flow_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--video_name", type=str)
    parser.add_argument("--destpath_flow", type=str)
    parser.add_argument("--suffix", type=str)

    args = parser.parse_args()

    process_video(video_path=args.filepath, video_name=args.video_name,  destpath_flow=args.destpath_flow)