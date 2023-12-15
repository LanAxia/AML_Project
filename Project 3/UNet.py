import torch
from torch import nn
import torch.nn.functional as F


# Unet模型
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super(DoubleConv, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(), 
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )

    def forward(self, x):
        output = self.sequence(x)
        return output


class DownModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownModel, self).__init__()
        self.sequence = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), 
            DoubleConv(in_channels, out_channels, mid_channels=out_channels)
        )

    def forward(self, x):
        output = self.sequence(x)
        return output


class UpModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super(UpModel, self).__init__()
        if not bilinear:
            self.up_model = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, padding=1, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=out_channels)
        else:
            self.up_model = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
    
    def forward(self, x1, x2):
        x1 = self.up_model(x1) # 此时x的形状可能和x2不一样
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - (diff_w // 2), diff_h // 2, diff_h - (diff_h // 2)]) # 将x1的形状和x2的形状一致

        output = torch.cat([x2, x1], dim=1)
        output = self.conv(output)
        return output


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        output = self.conv(x)
        return output
    

class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1 # 如果使用bilinear，up_model的输出和输入一致，为了连接之前down_model中的输出，要将上一次的输出改为原来的一半

        self.input_layer = DoubleConv(in_channels, 64, 64)
        self.down_model_1 = DownModel(64, 128)
        self.down_model_2 = DownModel(128, 256)
        self.down_model_3 = DownModel(256, 512)
        self.down_model_4 = DownModel(512, 1024 // factor)
        self.up_model_1 = UpModel(1024, 512 // factor, bilinear=bilinear)
        self.up_model_2 = UpModel(512, 256 // factor, bilinear=bilinear)
        self.up_model_3 = UpModel(256, 128 // factor, bilinear=bilinear)
        self.up_model_4 = UpModel(128, 64, bilinear=bilinear)
        self.output_layer = OutConv(64, out_channels)
    
    def forward(self, x):
        output_1 = self.input_layer(x)
        output_2 = self.down_model_1(output_1)
        output_3 = self.down_model_2(output_2)
        output_4 = self.down_model_3(output_3)
        output_5 = self.down_model_4(output_4)
        output = self.up_model_1(output_5, output_4)
        output = self.up_model_2(output, output_3)
        output = self.up_model_3(output, output_2)
        output = self.up_model_4(output, output_1)
        output = self.output_layer(output)
        return output
