from model import network
import math

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from einops import rearrange
import model.fusion_network as fusion


class Bell_loss(nn.Module):
    def __init__(self, gama, sita, *args, **kwargs) -> None:
        super(Bell_loss, self).__init__(*args, **kwargs)
        self.gama = gama
        self.sita = sita

    def forward(self, y_hat, y):
        loss = self.gama * \
            (1 - math.e ** (-(((y - y_hat)**2) / (2 * self.sita ** 2))))
        return torch.mean(loss)


class BIO_MODEL_LSTM(nn.Module):
    def __init__(self, arg, **args):
        super(BIO_MODEL_LSTM, self).__init__()

        self.cfg = arg

        self.audio_branch = network.get_audio_model(args=arg)

        self.video_branch = network.get_model(args=arg)
        # self.global_branch = network.get_model(args=arg)

        self.timemodel = network.get_time_model(args=arg)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.fusion_layer = fusion.Fusion(dim=512, depth=1, heads=8,
                                          dim_head=64, mlp_dim=2048, cfg=arg)

        self.linear2 = nn.Linear(64, 5)
        self.init_model()

    def init_model(self):
        torch.nn.init.normal_(self.linear2.weight.data, 0.1)
        if self.linear2.bias is not None:
            torch.nn.init.zeros_(self.linear2.bias.data)

    def parallel_extract(self, model, input):
        """
        batch, N, C, H, W --> batch*N, C, H, W
        then compute it by net
        batch*N, C, H, W --> batch, N, dim_feat 
        """
        B, N, c, h, w = input.shape
        input = rearrange(input, 'b n c h w -> (b n) c h w')
        feat = model(input)
        feat = rearrange(feat, '(b n) d -> b n d', b=B, n=N)

        return feat

    def slice_audio_features(self, input):
        """
        Divide the audio feature into N parts.
        """
        B, T, D = input.shape
        T_len = T // self.cfg.N
        outs = None
        for t in range(self.cfg.N):
            if t == self.cfg.N - 1:
                out = torch.sum(input[:, t*T_len:, :], dim=1, keepdim=True)
            else:
                out = torch.sum(input[:, t*T_len:(t+1)*T_len, :],
                                dim=1, keepdim=True)
            if outs is None:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=1)

        return F.relu(out)

    def forward(self, video_input, audio_input):
        audio_feat = self.audio_branch(audio_input)
        if not self.cfg.use_6MCFF:
            audio_feat = self.slice_audio_features(audio_feat)

        video_feat = self.parallel_extract(
            model=self.video_branch, input=video_input)
        # global_feat = self.parallel_extract(
        #     model=self.global_branch, input=global_input)

        # fusion_feat = torch.cat((video_feat, audio_feat), dim=2)
        fusion_feat = self.fusion_layer(video_feat, audio_feat)
        feat1 = self.dropout1(fusion_feat)

        # Don't use 6MCFF
        feat1 = self.timemodel(fusion_feat)

        # # use_6MCFF
        # feat1, _ = self.lstm1(feat1)
        # feat1, _ = self.lstm2(feat1)

        feat1 = self.dropout2(feat1)
        # feat2 = self.linear1(feat1)
        feat3 = self.linear2(feat1)
        feat4 = F.sigmoid(feat3)
        result = feat4.sum(dim=1) / self.cfg.N

        return result
