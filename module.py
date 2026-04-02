import torch
import torch.nn as nn
import torch.nn.functional as F


class CSJG(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.conv = nn.Conv2d(input_channel*2, input_channel, 1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, e, d):
        d = self.conv(d)
        mid_map = e + d
        map_relu = self.gelu(mid_map)
        attention_map = self.sigmoid(map_relu)
        return attention_map * e



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ASAM(nn.Module):
    def __init__(self, channels):
        super(ASAM, self).__init__()
        self.channels = channels
        dim = self.channels

        # 3*7 和 7*3 条带卷积 (捕捉方向性特征)
        self.fc_h = nn.Conv2d(dim, dim, (3, 7), stride=1, padding=(1, 7 // 2), groups=dim, bias=False)
        self.fc_w = nn.Conv2d(dim, dim, (7, 3), stride=1, padding=(7 // 2, 1), groups=dim, bias=False)

        self.channel_reweight = Mlp(dim, dim // 2, dim * 3)

        self.spatial_reweight = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 4, 3, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        N, C, H, W = x.shape
        x_w = self.fc_h(x)
        x_h = self.fc_w(x)

        # 聚合特征用于计算注意力
        x_add = x_h + x_w + x

        att_c = F.adaptive_avg_pool2d(x_add, output_size=1)
        att_c = self.channel_reweight(att_c).reshape(N, 3, C, 1, 1)
        att_c = F.silu(att_c)  # 形状: (N, 3, C, 1, 1)

        att_s = self.spatial_reweight(x_add)  # 形状: (N, 3, H, W)
        att_s = torch.sigmoid(att_s).unsqueeze(2)  # 形状: (N, 3, 1, H, W)

        x_att = (x_h * att_c[:, 0] * att_s[:, 0]) + \
                (x_w * att_c[:, 1] * att_s[:, 1]) + \
                (x * att_c[:, 2] * att_s[:, 2])

        return x_att



