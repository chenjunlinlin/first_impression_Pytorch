import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TDN_Net(nn.Module):

    def __init__(self, resnet_model, apha=0.5, belta=0.5, use_last_fc: bool = True):
        super(TDN_Net, self).__init__()

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)

        # implement conv1_5 and inflate weight
        self.conv1_temp = list(resnet_model.children())[0]
        params = [copy.deepcopy(x) for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * 4,) + kernel_size[2:]
        new_kernels = params[0].data.mean(
            dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5 = nn.Sequential(nn.Conv2d(
            12, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_5[0].weight.data = new_kernels

        self.maxpool_diff = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnext_layer1 = copy.deepcopy(
            nn.Sequential(*list(resnet_model.children())[4]))
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = list(resnet_model.children())[8]
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        if use_last_fc:
            self.fc = list(resnet_model.children())[9]
        self.apha = apha
        self.belta = belta
        self.use_last_fc = use_last_fc

    def forward(self, x):
        x1, x2, x3, x4, x5 = x[:, 0:3, :, :], x[:, 3:6, :,
                                                :], x[:, 6:9, :, :], x[:, 9:12, :, :], x[:, 12:15, :, :]
        x_c5 = self.conv1_5(self.avg_diff(torch.cat(
            [x2-x1, x3-x2, x4-x3, x5-x4], 1).view(-1, 12, x2.size()[2], x2.size()[3])))
        x_diff = self.maxpool_diff(1.0/1.0*x_c5)  # don't change size

        temp_out_diff1 = x_diff
        x_diff = self.resnext_layer1(x_diff)

        x = self.conv1(x3)
        x = self.bn1(x)
        x = self.relu(x)
        # fusion layer1
        x = self.maxpool(x)
        temp_out_diff1 = F.interpolate(temp_out_diff1, x.size()[2:])
        x = self.apha*x + self.belta*temp_out_diff1
        # fusion layer2
        x = self.layer1_bak(x)
        x_diff = F.interpolate(x_diff, x.size()[2:])
        x = self.apha*x + self.belta*x_diff

        x = self.layer2_bak(x)
        x = self.layer3_bak(x)
        x = self.layer4_bak(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.use_last_fc:
            x = self.fc(x)

        return x
