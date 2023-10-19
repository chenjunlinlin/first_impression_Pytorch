from utils import extract_frames, extract_flows
import multiprocessing
import tqdm
import os
import pandas as pd

resource_path = "/raid5/chenjunlin/DataSets/first_impression/"
destpath_rgb = ["trainframes_face2", "validationframes_face2"]
destpath_global = ["trainframes_global", "validationframes_global"]
resource_flow = ["train_flow", "validation_flow"]
destpath_flow = ["trainface_flow2", "validationface_flow2"]
suffix = ".jpg"
add_flow_path = []

def process_video(videos_path, i):
    if i == 0:
        for video_path in tqdm.tqdm(videos_path, desc="extract frames:"):
            video_name = os.path.split(video_path)[-1]
            extract_frames.process(vid_path=video_path, 
                                video_name=video_name,
                                resource_path=resource_path,
                                destpath_rgb=destpath_rgb,
                                destpath_global = destpath_global,
                                resource_flow=resource_flow,
                                destpath_flow=destpath_flow,
                                suffix=suffix
                                )
    else:
        for video_path in videos_path:
            video_name = os.path.split(video_path)[-1]
            extract_frames.process(vid_path=video_path, 
                                video_name=video_name,
                                resource_path=resource_path,
                                destpath_rgb=destpath_rgb,
                                destpath_global = destpath_global,
                                resource_flow=resource_flow,
                                destpath_flow=destpath_flow,
                                suffix=suffix
                                )
def process_flow(videos_path, i):
    if i == 0:
        for video_path in tqdm.tqdm(videos_path, desc='extract flows:'):
            video_name = os.path.split(video_path)[-1]
            extract_flows.process(video_path=video_path,
                                   video_name=video_name, 
                                   resource_path=resource_path, 
                                   add_flow_path=add_flow_path)
    else:
        for video_path in videos_path:
            video_name = os.path.split(video_path)[-1]
            extract_flows.process(video_path=video_path,
                                   video_name=video_name, 
                                   resource_path=resource_path, 
                                   add_flow_path=add_flow_path)


if __name__ == "__main__":

    N = 2

    # videos_list = pd.read_csv("dataset/val.csv")["VideoName"]
    videos_list = ['./dataset/train/training80_31/syTTeox8Yaw.003.mp4', './dataset/train/training80_59/xyWpSrfFlQw.004.mp4']
    processes = []
    for i in range(N):
        per_len = len(videos_list) // N
        if i == (N - 1):
            part_loses_path = videos_list[i*per_len:]
        else:
            part_loses_path = videos_list[i*per_len:(i+1)*per_len]
        process = multiprocessing.Process(target=process_video, args=(part_loses_path, i))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()