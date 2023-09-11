import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)

import network
import ViT


class MY_model(nn.Module):
    def __init__(self, args, **kwargs):
        super(MY_model, self).__init__()

        self.args = args
        self.vid_branch = network.get_model(
            outputdim=240, pretrained= not self.args.pretrain, progress=True, model_name=self.args.model_name)
        self.aud_branch = nn.Linear(68, 32)
        # self.flow_branch = network.get_model(
        #     outputdim=160, pretrained= not self.args.pretrain, progress=True, model_name=self.args.model_name)
        self.ViT = ViT.ViT(num_patches=self.args.N, num_classes=5, dim=272, 
                           depth=12,heads=6, dim_head=72, mlp_dim=864)
        
    def forward(self, vid_input, flow_input, aud_input):
        aud_feat = self.aud_branch(aud_input)
        vid_feat = torch.zeros(0).cuda()
        flow_feat = torch.zeros(0).cuda()
        for i in range(self.args.N):
            feat1 = self.vid_branch(vid_input[:, i, ...].squeeze(1))
            if i == 0:
                vid_feat = feat1.unsqueeze(1)
            else:
                vid_feat = torch.cat((vid_feat, feat1.unsqueeze(1)), dim=1)
        # for i in range(self.args.N):
        #     feat2 = self.flow_branch(flow_input[:, i, ...].squeeze(1))
        #     if i == 0:
        #         flow_feat = feat2.unsqueeze(1)
        #     else:
        #         flow_feat = torch.cat((flow_feat, feat2.unsqueeze(1)),dim=1)

        # fusion_feat = torch.cat((aud_feat, vid_feat, flow_feat), dim=2)
        fusion_feat = torch.cat((aud_feat, vid_feat), dim=2)
        result = F.sigmoid(self.ViT(fusion_feat))

        return result