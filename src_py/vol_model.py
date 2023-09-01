import torch
import torch.nn as nn


class VOL_MODEL(nn.Module):
    def __init__(self, **args):
        super(VOL_MODEL, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 16, (3, 5, 5))
        self.norm1 = nn.BatchNorm3d(16)
        self.maxpool1 = nn.MaxPool3d(2, 2)
        self.conv3d_2 = nn.Conv3d(16, 16, (2, 5, 5))
        self.norm2 = nn.BatchNorm3d(16)
        self.maxpool2 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.conv3d_3 = nn.Conv3d(16, 1, (1, 5, 5))
        self.norm3 = nn.BatchNorm3d(1)
        self.drop = nn.Dropout(0.2)
        self.linear1 = nn.Linear(441, 200)
        self.linear2 = nn.Linear(200, 5)

    def forward(self, video_input, video_name):
        feat1 = self.norm1(self.conv3d_1(video_input))
        feat2 = nn.ReLU()(feat1)
        feat3 = self.maxpool1(feat2)
        feat4 = self.norm2(self.conv3d_2(feat3))
        feat5 = nn.ReLU()(feat4)
        feat6 = self.maxpool2(feat5)
        feat7 = self.norm3(self.conv3d_3(feat6))
        feat8 = nn.Flatten()(feat7)
        feat9 = self.drop(feat8)
        feat10 = self.linear1(feat9)
        feat11 = nn.ReLU()(feat10)
        feat12 = self.drop(feat11)
        feat13 = self.linear2(feat12)
        result = nn.Sigmoid()(feat13)

        return result