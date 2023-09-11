import cv2
import numpy as np
import os
import dlib
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if os.path.isdir(path):
            pass
        else:
            raise

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

def compute_TVL1(prev, curr, bound=15):
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)

    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def get_image_face(img, flow):

    face_detector = dlib.get_frontal_face_detector()
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces):
        # For now only take biggest face
        face = faces[0]

        # --- Prediction ---------------------------------------------------
        # Face crop with dlib and bounding box scale enlargement
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = img[y:y+size, x:x+size]
        cropped_flow = flow[y:y+size, x:x+size]
        return cropped_face, cropped_flow
    return None, None

def save_image(destpath, vid_name, num, image, suffix):
    vid_name = vid_name.replace(".mp4", '')
    mkdir_p(os.path.join(destpath,vid_name))
    file_name = os.path.join(destpath, vid_name , vid_name + "_{:04d}{}".format(num, suffix))
    # print(file_name)
    cv2.imwrite(file_name, image)

def save_flow(destpath, vid_name, num, image, suffix):
    vid_name = vid_name.replace(".mp4", '')
    mkdir_p(os.path.join(destpath,vid_name, "{:04d}".format(num)))
    pre_file_name = os.path.join(destpath, vid_name , "{:04d}".format(num), "pre.{}".format(suffix))
    cur_file_name = os.path.join(destpath, vid_name , "{:04d}".format(num), "cur.{}".format(suffix))
    # print(file_name)
    cv2.imwrite(pre_file_name, image[:, :, 0])
    cv2.imwrite(cur_file_name, image[:, :, 1])

def resize_img(frame):
    frame = torch.tensor(frame)
    frame = frame.permute(2, 0, 1)
    frame = F.resize(frame, [520, 960], antialias=False)
    frame = frame.permute(1, 2, 0)
    frame = frame.numpy()
    frame.dtype = np.uint8

    return frame

def process_video(vid_path, video_name, destpath_rgb, resource_flow, destpath_flow, suffix):
    videoCapture = cv2.VideoCapture(vid_path)
    if video_name[0] == "*":
        video_name = '-' + video_name[1:]

    video_name = video_name.replace(".mp4", "")
    flows_path = os.path.join(resource_flow, video_name)
    i = 0
    while True:
        success, frame = videoCapture.read()
        if success:
            if i > 0:
                flow_path = os.path.join(flows_path, f"{(i-1):04d}.jpg")
                flow = cv2.imread(filename=flow_path)
                frame = resize_img(frame=frame)
                img, flow = get_image_face(frame, flow=flow)
                if img.all() != None:
                    save_image(destpath_rgb, video_name, i, img, suffix=suffix)
                    save_image(destpath_flow, video_name, i, flow, suffix=suffix)
            i = i + 1
        else:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--video_name", type=str)
    parser.add_argument("--destpath_rgb", type=str)
    parser.add_argument("--resource_flow", type=str)
    parser.add_argument("--destpath_flow", type=str)
    parser.add_argument("--suffix", type=str)

    args = parser.parse_args()

    process_video(vid_path=args.filepath, video_name=args.video_name, destpath_rgb=args.destpath_rgb, resource_flow=args.resource_flow, destpath_flow=args.destpath_flow, suffix=args.suffix)