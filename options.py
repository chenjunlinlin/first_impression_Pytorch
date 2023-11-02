import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--train_csv_path',
                    default="./dataset/training_gt.csv", type=str,
                    help='训练数据的csv文件路径')
parser.add_argument('--val_csv_path',
                    default='./dataset/val_gt.csv',
                    help='测试数据的csv文件路径')
parser.add_argument('--train_audio_dir',
                    default='./dataset/trainaudiofeat', type=str,
                    help='训练音频目录')
parser.add_argument('--val_audio_dir',
                    default='./dataset/validationaudiofeat',
                    type=str, help='测试音频目录')
parser.add_argument('--train_video_dir',
                    default='./dataset/trainframes_face2',
                    type=str, help='训练视频路径')
parser.add_argument('--train_flow_dir', default='./dataset/train_flow2',
                    type=str, help='训练视频路径')
parser.add_argument('--train_global_dir',
                    default='./dataset/trainframes_global',
                    type=str, help='训练视频全局图片路径')
parser.add_argument('--val_video_dir',
                    default='./dataset/validationframes_face2',
                    type=str, help='测试视频路径')
parser.add_argument('--val_flow_dir', default='./dataset/validation_flow2',
                    type=str, help='测试视频路径')
parser.add_argument('--val_global_dir',
                    default='./dataset/validationframes_global',
                    type=str, help='验证视频全局图片路径')
parser.add_argument('--best_model_save_dir',
                    default='./models/BestModel', type=str,
                    help='最优模型路径')
parser.add_argument('--model_save_dir', default='./models/checkpoints/',
                    type=str, help='模型保存地址')
parser.add_argument('--model_path', default=None,
                    type=str, help='模型地址，可指定模型权重')


parser.add_argument('--N', default=6, type=int, help='视频被分成的份数')
parser.add_argument('--num_threads', default=2,
                    type=int, help='加载frames的线程数')
parser.add_argument("--devices", default=[0, 1, 2, 3])
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--opt', default="SGD", choices=["SGD", "Adam"])
parser.add_argument('--scheduler', default="StepLR",
                    choices=["StepLR", "CosineAnnealingWarmRestarts"])
parser.add_argument('--lr', default=0.05)
parser.add_argument('--num_flow', default=3, help="num of flows per frame")
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--weight_decay', default=1e-3)
parser.add_argument('--epochs', default=300)
parser.add_argument('--iscuda', default=True)
parser.add_argument('--pretrain', default=True)
parser.add_argument("--_continue", default=False)
parser.add_argument('--backbone', default='resnet34', type=str)
parser.add_argument('--gama', default=0)
parser.add_argument('--sita', default=9)
parser.add_argument('--img_size', default=224)
parser.add_argument('--dim_img', default=200)
parser.add_argument('--dim_audio', default=32)
parser.add_argument("--seed", default=3407)
parser.add_argument("--use_6MCFF", default=False)
parser.add_argument("--DP", default=True)

parser.add_argument('--name', default='exp')
parser.add_argument(
    '--dsc', default='本次实验测试，face+encoder——audio+trans——fusion+trans——time')

parser.add_argument('--logs', type=str, default='./logs')

args = parser.parse_args()


def get_args():
    return parser.parse_args()
