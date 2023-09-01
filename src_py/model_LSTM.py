import torch
import torch.nn as nn
import torch.nn.functional as F


class BIO_MODEL_LSTM(nn.Module):
    def __init__(self, **args):
        super(BIO_MODEL_LSTM, self).__init__()

        # audio_branch
        self.audio_branch =nn.Sequential(
            nn.Linear(68, 32)
            )  # -- 6x68 --> 6x32

        # vedio_branch
        self.video_branch = nn.Sequential()
        self.video_branch.append(self.video_branch_layer(3, 16, (1,5,5), (1,2,2)))
        self.video_branch.append(self.video_branch_layer(16, 16, (1,7,7), (1,2,2)))
        self.video_branch.append(self.video_branch_layer(16, 16, (1,9,9), (1,2,2)))

        # self.dropout1 = nn.Dropout(0.2)
        # self.dropout2 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(160, 128, batch_first=True)
        self.linear1 = nn.Linear(16*8*8, 128)
        self.linear2 = nn.Linear(128, 5)

    def video_branch_layer(self, input_channels, output_channels, kernel_size, pool_kernel_size):
        layer = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size),
            nn.ReLU(),
            nn.MaxPool3d(pool_kernel_size)
        )
        return layer

    def forward(self, video_input, audio_input):
        audio_feat = self.audio_branch(audio_input)
        video_feat1 = self.video_branch(video_input)
        video_feat2 = video_feat1.transpose(1,2).flatten(start_dim=2)
        video_feat3 = self.linear1(video_feat2)
        video_feat = nn.ReLU()(video_feat3)
        fusion_feat = torch.cat((audio_feat, video_feat), 2)

        # feat1 = self.dropout(fusion_feat)
        feat2, _ = self.lstm(fusion_feat)
        # feat3 = self.dropout(feat2)
        feat4 = self.linear2(feat2)
        feat5 = F.sigmoid(feat4)
        result = feat5.sum(1) / 6
        return result


# if __name__ == '__main__':
#     audio_input = torch.rand((6, 68))
#     video_input = torch.rand((6, 3, 112, 112))
#     bio_model = BIO_MODEL_LSTM()
#     result = bio_model(audio_input, video_input)
#     print(result)
