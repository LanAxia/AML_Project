import torch
from torch import nn

class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=17, stride=2, padding=8):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        elif stride != 1: # Channel想通但是有stride
            self.shortcut = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        output = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        residual = self.shortcut(x)
        output = output + residual
        return output


class MyModel(nn.Module):
    def __init__(self, input_channels: int = 1, output_features: int = 32, output_dim: int = 4):
        super(MyModel, self).__init__()

        self.layer_1 = nn.Sequential( # 未使用残差的CNN
            nn.Conv1d(input_channels, 64, kernel_size=16, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential( # 8组残差视觉网络
            Residual_Block(64, 64, kernel_size=17, stride=2, padding=8), 
            Residual_Block(64, 64, kernel_size=17, stride=2, padding=8), 
            Residual_Block(64, 128, kernel_size=17, stride=2, padding=8), 
            Residual_Block(128, 128, kernel_size=17, stride=2, padding=8), 
            Residual_Block(128, 256, kernel_size=17, stride=2, padding=8), 
            Residual_Block(256, 256, kernel_size=17, stride=2, padding=8), 
            Residual_Block(256, 512, kernel_size=17, stride=2, padding=8), 
            Residual_Block(512, 512, kernel_size=17, stride=2, padding=8), 
        )

        self.layer_3 = nn.Sequential( # BiLSTM
            nn.BatchNorm1d(512), 
            nn.ReLU(), 
            nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True), 
        )

        self.layer_4 = nn.Sequential( # 解析为32维向量
            nn.Dropout(0.5), 
            nn.Linear(512, output_features), 
        )

        self.layer_5 = nn.Sequential( # 
            nn.Dropout(0.5), 
            nn.Linear(output_features, output_dim)
        )


    def embed(self, x):
        output = self.layer_1(x)
        output = self.layer_2(output)
        output, (h_n, c_n) = self.layer_3(output)
        output = self.layer_4(output[:, -1, :])
        return output
    
    def forward(self, x):
        output = self.embed(x)
        output = self.layer_5(output)
        return output