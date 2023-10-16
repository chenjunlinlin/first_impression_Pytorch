from utils import extract_frames, extract_flows
import multiprocessing
import tqdm
import os

resource_path = "/raid5/chenjunlin/DataSets/first_impression/"
destpath_rgb = ["lose_trainframes_face", "lose_validationframes_face"]
resource_flow = ["lose_train_flow", "lose_validatin_flow"]
destpath_flow = ["lose_trainface_flow", "lose_validationface_flow"]
suffix = ".jpg"

add_flow_path = ["lose_train_flow", "lose_validatin_flow"]

def process_video(videos_path, i):
    if i == 0:
        for video_path in tqdm.tqdm(videos_path, desc="extract frames:"):
            video_name = os.path.split(video_path)[-1]
            extract_frames.process(vid_path=video_path, 
                                video_name=video_name,
                                resource_path=resource_path,
                                destpath_rgb=destpath_rgb,
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

    loses_path = []
    N = 12
    N2 = 2

    with open("/raid5/chenjunlin/first-impressions-master/dataset/lose.txt", 'r') as f:
        loses = f.readline()
        loses_path = loses.split(',')[:-1]
    
    print("未处理完成的视频共有：", len(loses_path))

    # processes_flow = []
    # for i in range(N2):
    #     per_len = len(loses_path) // N2
    #     if i == (N2 - 1):
    #         part_loses_path = loses_path[i*per_len:]
    #     else:
    #         part_loses_path = loses_path[i*per_len:(i+1)*per_len]
    #     process = multiprocessing.Process(target=process_flow, args=(part_loses_path, i))
    #     processes_flow.append(process)
    #     process.start()
    # for process in processes_flow:
    #     process.join()

    processes = []
    for i in range(N):
        per_len = len(loses_path) // N
        if i == (N - 1):
            part_loses_path = loses_path[i*per_len:]
        else:
            part_loses_path = loses_path[i*per_len:(i+1)*per_len]
        process = multiprocessing.Process(target=process_video, args=(part_loses_path, i))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()