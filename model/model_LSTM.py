import torch
import torch.nn as nn
import os

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
        self.audio_branch =nn.Sequential(
            nn.Linear(68, 32)
            )  # -- 6x68 --> 6x32

        self.vid_branch = network.get_model(outputdim=200, pretrained= not arg.pretrain, progress=True, model_name=arg.backbone)
        self.flow_branch = network.get_model(outputdim=200, pretrained= not arg.pretrain, progress=True, model_name=arg.backbone)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.lstm1 = nn.LSTM(432, 221, batch_first=True)
        # self.lstm2 = nn.LSTM(221, 64, batch_first=True)
        self.pos_embedding = nn.Parameter(torch.randn(1, arg.N, 512))
        self.trans_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.5, dim_feedforward=2048, batch_first=True)
        self.timemodel = nn.TransformerEncoder(encoder_layer=self.trans_layer, num_layers=3)
        self.embeding = nn.Linear(232, 512)
        self.aux_linear = nn.Linear(512, 5)
        self.linear1 = nn.Linear(512, 224)
        self.linear2 = nn.Linear(224, 64)
        self.linear3 = nn.Linear(64, 5)
        self.init_model()

			 
    def init_model(self):
        for m in self.audio_branch:
            torch.nn.init.normal_(m.weight.data, 0.1)
            if m.bias is not None:
                  torch.nn.init.zeros_(m.bias.data)
        torch.nn.init.normal_(self.linear2.weight.data, 0.1)
        if self.linear2.bias is not None:
            torch.nn.init.zeros_(self.linear2.bias.data)
                            

    def forward(self, video_input, audio_input):
        audio_feat = self.audio_branch(audio_input)

        batch, N, C, H, W = video_input.shape[0], video_input.shape[1], video_input.shape[2], video_input.shape[3], video_input.shape[4]
        video_feat = video_input.reshape((batch*N, C, H, W))
        video_feat = self.vid_branch(video_feat)
        dim_vid = video_feat.shape[-1]
        video_feat = video_feat.reshape((batch, N, dim_vid))
        
        # flow_feat = flow_input.reshape((batch*N, C, H, W))
        # flow_feat = self.flow_branch(flow_feat)
        # dim_flow = flow_feat.shape[-1]
        # flow_feat = flow_feat.reshape((batch, N, dim_flow))
        
        fusion_feat = torch.cat((audio_feat, video_feat), 2)
        feat1 = self.dropout1(fusion_feat)
        src = self.embeding(feat1)
        src += self.pos_embedding
        feat2 = self.timemodel(src)
        feat3 = self.dropout2(feat2)
        feat4 = self.linear1(feat3)
        feat5 = self.linear2(feat4)
        feat6 = self.linear3(feat5)
        feat7 = nn.Sigmoid()(feat6)
        aux_feat = self.aux_linear(feat2)
        aux_feat = nn.Sigmoid()(aux_feat)
        result = feat7.sum(1) / 6
        aux_result = aux_feat.sum(1) / 6

        return result, aux_result


