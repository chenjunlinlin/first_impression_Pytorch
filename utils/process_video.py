from extract_frames import *
from extract_flows import *
import subprocess
import tqdm


video_root = ['train']
out_root = ['../dataset/trainframes_face']
suffix = '.jpg'
flow_root = ['../dataset/train_flow']
flow_face_root = ['../dataset/trainface_flow']

def extract_frames(filepath, suffix, index):

    if(os.path.isfile(filepath)):
        if(filepath.lower().endswith('.mp4')):
            path, filename = os.path.split(filepath)
            destpath_rgb =  os.path.abspath(out_root[index])
            resource_flow = os.path.abspath(flow_root[index])
            destpath_flow =  os.path.abspath(flow_face_root[index])
            mkdir_p(destpath_rgb)
            mkdir_p(destpath_flow)
            if filename.startswith('-'):
                print(filename)
                filename = "*" + filename[1:]
                print(filename)
            command = "python " + "extract_frames.py " + "--filepath " + filepath + " --video_name " + filename +" --destpath_rgb " + destpath_rgb + " --resource_flow " + resource_flow + " --destpath_flow " + destpath_flow + " --suffix " + suffix
            subprocess.call(command, shell=True)
    else:
        allfiles = [f for f in os.listdir(filepath) if (f != '.' and f != '..')]
        for anyfile in allfiles:
            extract_frames(os.path.join(filepath, anyfile), suffix=suffix, index=index)


def extract_flows(filepath, suffix, index):

    if(os.path.isfile(filepath)):
        if(filepath.lower().endswith('.mp4')):
            path, filename = os.path.split(filepath)
            destpath_flow =  os.path.abspath(flow_root[index])
            mkdir_p(destpath_flow)
            if filename.startswith('-'):
                filename = "*" + filename[1:]
            command = "python " + "extract_flows.py " + "--filepath " + filepath + " --video_name " + filename  + " --destpath_flow " + destpath_flow + " --suffix " + suffix
            subprocess.call(command, shell=True)
    else:
        allfiles = [f for f in os.listdir(filepath) if (f != '.' and f != '..')]
        for anyfile in allfiles:
            extract_flows(os.path.join(filepath, anyfile), suffix=suffix, index=index)

def main():
    for i , dir in enumerate(video_root):
        videos_path = os.path.join('../dataset' , dir)
        for video_path in tqdm.tqdm(os.listdir(videos_path), desc=videos_path):
            if video_path != '__MACOSX' and video_path != '.ipynb_checkpoints':
                extract_frames(os.path.join(videos_path, video_path), suffix, i)


if __name__ == '__main__':
    main()
    print("finish!!!!!!!!!!!!!!!!!!")