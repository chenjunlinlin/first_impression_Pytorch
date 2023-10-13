import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--train_csv_path', default="./dataset/training_gt.csv", type=str, help='训练数据的csv文件路径')
parser.add_argument('--val_csv_path', default='./dataset/val_gt.csv',    help='测试数据的csv文件路径')
parser.add_argument('--train_audio_dir', default='./dataset/trainaudiofeat', type=str, help='训练音频目录')
parser.add_argument('--val_audio_dir', default='./dataset/validationaudiofeat', 
                    type=str, help='测试音频目录')
parser.add_argument('--train_video_dir', default='./dataset/trainframes', 
                    type=str, help='训练视频路径')
parser.add_argument('--train_flow_dir', default='./dataset/train_flow', 
                    type=str, help='训练视频路径')
parser.add_argument('--val_video_dir', default='./dataset/validationframes', 
                    type=str, help='测试视频路径')
parser.add_argument('--val_flow_dir', default='./dataset/validation_flow', 
                    type=str, help='测试视频路径')
parser.add_argument('--best_model_save_dir', default='./models/BestModel', type=str, help='最优模型路径')
parser.add_argument('--model_save_dir', default='./models/BioModel/', type=str, help='模型保存地址')


parser.add_argument('--N', default=6, type=int, help='视频被分成的份数')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--lr', default=0.05)
parser.add_argument('--num_flow', default=3, help="num of flows per frame")
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--weight_decay', default=5e-3)
parser.add_argument('--epochs', default=300)
parser.add_argument('--iscuda', default=True)
parser.add_argument('--pretrain', default=False)
parser.add_argument('--backbone', default='resnet18', type=str)
parser.add_argument('--gama', default=300)
parser.add_argument('--sita', default=9)

parser.add_argument('--name', default='resnet18_2stream_3translay_96bs_consi_belloss_4', type=str)
parser.add_argument('--logs', type=str, default='./logs')

args = parser.parse_args()

def get_args():
    return parser.parse_args()