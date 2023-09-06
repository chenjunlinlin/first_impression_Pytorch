import network
import torch
import torch.nn as nn

class MY_modle(nn.modules):
    def __init__(self, pretrained, N, **kwargs):
        super(MY_modle, self).__init__()
    
        self.vid_branch = network.resnet18(pretrained=True, progress=True)
        self.aud_branch = nn.Sequential(
            nn.Linear(136, 68),
            nn.Linear(68, 32)
        )
        self.flow_branch = network.resnet18(pretrained=True, progress=True)
        self.linear_vid = nn.Linear(512, 240)
        self.linear_flow = nn.Linear(512, 160)
        self.N = N
        

    
    def forward(self, vid_input, flow_input, aud_input):
        aud_feat = self.aud_branch(aud_input)
        vid_feat = torch.zeros(0)
        for i in range(self.N):
            feat1 = self.vid_branch(vid_input[:, i, ...].squeeze(1))
            if i == 0:
                vid_feat = feat1.unsqueeze(1)
            else:
                vid_feat = torch.cat((vid_feat, feat1.unsqueeze(1)))
        flow_feat = torch.zeros(0)
        for i in range(self.N):
            feat2 = self.flow_branch(flow_input[:, i, ...].squeeze(1))
            if i == 0:
                flow_feat = feat2.unsqueeze(1)
            else:
                flow_feat = torch.cat((flow_feat, feat2.unsqueeze(1)))

        fusion_feat = torch.cat((aud_feat, vid_feat, flow_feat), dim=2)