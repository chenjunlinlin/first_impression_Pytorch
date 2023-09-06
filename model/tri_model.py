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
        vid_feat = []
        