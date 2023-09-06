from extract_frames import *
import subprocess
import tqdm


video_root = ['train', 'test', 'validation']
out_root = ['../dataset/trainframes_origin', '../dataset/testframes_origin', '../dataset/validationframes_origin']
suffix = '.png'
flow_root = ['../dataset/trainframes_flow', '../dataset/testframes_flow', '../dataset/validationframes_flow']

def extract_frames(filepath, suffix, index):

    if(os.path.isfile(filepath)):
        if(filepath.lower().endswith('.mp4')):
            path, filename = os.path.split(filepath)
            print("out_root_path: {}".format(out_root[index]))
            destpath =  os.path.abspath(out_root[index])
            print("destpath: {}".format(destpath))
            mkdir_p(destpath)
            command = "python " + "extract_frames.py " + "--filepath " + filepath + " --video_name " + filename +" --destpath " + destpath + " --suffix " + suffix
            print(command)
            # process(filepath, video_name=filename, destpath=destpath, suffix=suffix)
            subprocess.call(command, shell=True)
            print("extract frmae from {}".format(filepath))
    else:
        allfiles = [f for f in os.listdir(filepath) if (f != '.' and f != '..')]
        for anyfile in allfiles:
            print(os.path.join(filepath, anyfile))
            extract_frames(os.path.join(filepath, anyfile), suffix=suffix, index=index)

def main():
    for i , dir in enumerate(video_root):
        videos_path = os.path.join('../dataset' , dir)
        for video_path in tqdm.tqdm(os.listdir(videos_path), desc=videos_path):
            if video_path != '__MACOSX' and video_path != '.ipynb_checkpoints':
                extract_frames(os.path.join(videos_path, video_path), suffix, i)


if __name__ == '__main__':
    main()
    print("finish!!!!!!!!!!!!!!!!!!")