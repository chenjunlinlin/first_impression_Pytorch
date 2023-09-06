import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = [0]
parser = argparse.ArgumentParser()

# 文件信息
parser.add_argument('--train_csv_path', default="./data/training_gt.csv", type=str, help='训练数据的csv文件路径')
parser.add_argument('--val_csv_path', default='./data/val_gt.csv', help='测试数据的csv文件路径')
parser.add_argument('--train_audio_dir', default='./data/trainaudiofeat', type=str, help='训练音频目录')
parser.add_argument('--val_audio_dir', default='./data/validationaudiofeat', 
                    type=str, help='测试音频目录')
parser.add_argument('--train_video_dir', default='./data/trainframes', 
                    type=str, help='训练视频路径')
parser.add_argument('--val_video_dir', default='./data/validationframes', 
                    type=str, help='测试视频路径')
parser.add_argument('--best_model_save_dir', default='./models/BestModel', type=str, help='最优模型路径')
parser.add_argument('--model_save_dir', default='./models/BioModel', type=str, help='模型保存地址')

# 训练参数
parser.add_argument('--N', default=6, type=int, help='视频被分成的份数')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.05)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--weight_decay', default=5e-3)
parser.add_argument('--epochs', default=100)
parser.add_argument('--iscuda', default=False)
parser.add_argument('--pretrain', default=True)

# 实验日志
parser.add_argument("--name", type=str, help='实验名称')
parser.add_argument("--best_model", type=str, default="./logs/best_model.json")

def get_args():
    return parser.parse_args()