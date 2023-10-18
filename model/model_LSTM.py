import torch
import torch.nn as nn
import os
import torch.nn.functional as F

import sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
import network
import math


class Bell_loss(nn.Module):
    def __init__(self, gama, sita, *args, **kwargs) -> None:
        super(Bell_loss, self).__init__(*args, **kwargs)
        self.gama = gama
        self.sita = sita

    def forward(self, y_hat, y):
        loss = self.gama * (1 - math.e ** (-(((y - y_hat)**2) / (2* self.sita ** 2))))
        return torch.mean(loss)
    
class BIO_MODEL_LSTM(nn.Module):
    def __init__(self, arg, **args):
        super(BIO_MODEL_LSTM, self).__init__()

        # audio_branch
        self.audio_branch = network.get_audio_model()


        self.video_branch = network.get_model(args=arg)

        self.timemodel = network.get_time_model(args=arg)

        self.dropout1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(arg.dim_img+arg.dim_audio, 128)
        self.linear2 = nn.Linear(128, 5)

    def parallel_extract(self, net, input):
        """
        batch, N, C, H, W --> batch*N, C, H, W
        then compute it by net
        batch*N, C, H, W --> batch, N, dim_feat 
        """
        batch, N, C, H, W = input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4]
        feat = input.reshape((batch*N, C, H, W))
        feat = net(feat)
        dim_feat = feat.shape[-1]
        feat = feat.reshape((batch, N, dim_feat))

        return feat

    def forward(self, video_input, audio_input):
        audio_feat = self.audio_branch(audio_input)

        video_feat = self.parallel_extract(net=self.video_branch,
                                           input=video_input)
        
        video_feat = self.timemodel(video_feat)
        fusion_feat = torch.cat((video_feat, audio_feat), dim=2)
        feat1 = self.dropout1(fusion_feat)
        feat2 = self.linear1(feat1)
        feat3 = self.linear2(feat2)
        feat3 = torch.squeeze(feat3)
        result = F.softmax(feat3)

        return result


