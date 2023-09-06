import cv2
import numpy as np
import os
import dlib
import argparse


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

def get_image_face(img):

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
        return cropped_face

def save_image(destpath, vid_name, num, image, suffix):
    vid_name = vid_name.replace(".mp4", '')
    mkdir_p(os.path.join(destpath,vid_name))
    file_name = os.path.join(destpath, vid_name , vid_name + "_{:04d}{}".format(num, suffix))
    # print(file_name)
    cv2.imwrite(file_name, image)


def process_video(vid_path, video_name, destpath, suffix):
    videoCapture = cv2.VideoCapture(vid_path)

    i = 0
    while True:
        success, frame = videoCapture.read()
        if success:
            i = i + 1
            # img = get_image_face(frame)
            save_image(destpath, video_name, i, frame, suffix=suffix)
        else:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--video_name", type=str)
    parser.add_argument("--destpath", type=str)
    parser.add_argument("--suffix", type=str)

    args = parser.parse_args()

    process_video(vid_path=args.filepath, video_name=args.video_name, destpath=args.destpath, suffix=args.suffix)