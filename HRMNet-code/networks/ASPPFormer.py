import torch
import torch.nn as nn
from timm.layers import DropPath


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        return self.relu(self.bn(self.conv(x)))

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class BasicASConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = AtrousSeparableConvolution(in_planes, out_planes, kernel_size=kernel_size,stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        return self.relu(self.bn(self.conv(x)))        

class ConvMlp(nn.Module):
    def __init__(self, in_features=64, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features*4
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=1),
            act_layer(),
            nn.Conv2d(hidden_features, out_features, 1),
            DropPath(0.1),
        )

    def forward(self, x):
        return self.conv_mlp(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class LayerNorm2d(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class ASPPFormer(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256, dilation_1=3, dilation_2=2, channel_up=None):
        """channel_up: x_up 的通道数，用于 conv3 投影。若 None 且 channel_2==channel_1 则用 channel_1//2"""
        super().__init__()
        self.conv1 = BasicConv2d(channel_1, channel_1//2, 3, padding=1)
        self.conv1_Dila = BasicASConv2d(channel_1//2, channel_3, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_3, 3, padding=1)
        self.conv2_Dila = BasicASConv2d(channel_3, channel_3, 3, padding=dilation_2, dilation=dilation_2)
        
        self.conv_mlp = ConvMlp(channel_3)
        self.drop = DropPath(0.1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(channel_3, channel_3, kernel_size=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        if channel_up is not None:
            self.conv3 = BasicConv2d(channel_up, channel_3)
        elif channel_2 == channel_1:
            self.conv3 = BasicConv2d(channel_1 // 2, channel_3)
        else:
            self.conv3 = nn.Identity()
        
        self._init_weights()

    def forward(self, x, x_up=None):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)
    
        x2 = self.conv2(torch.cat((x1,x_up),dim=1) if x_up is not None else x1)
        x2_dila = self.conv2_Dila(x2) 

        x_fuse = self.conv3(x_up) + x1_dila + self.drop(self.conv_mlp(x2_dila)) if x_up is not None else x1_dila + self.drop(self.conv_mlp(x2_dila))
        return self.conv_last(x_fuse)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



