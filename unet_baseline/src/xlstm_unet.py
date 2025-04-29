import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这一行导入F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        # Split gates into different components
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class xLSTMUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_c=32):
        super(xLSTMUNet, self).__init__()
        self.base_c = base_c

        # Encoder path
        self.enc1 = DoubleConv(in_channels, base_c)
        self.lstm1 = ConvLSTMCell(base_c, base_c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_c, base_c * 2)
        self.lstm2 = ConvLSTMCell(base_c * 2, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_c * 2, base_c * 4)
        self.lstm3 = ConvLSTMCell(base_c * 4, base_c * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base_c * 4, base_c * 8)
        self.lstm4 = ConvLSTMCell(base_c * 8, base_c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_c * 8, base_c * 16)

        # Decoder path - 修改转置卷积参数，添加output_padding以处理奇数尺寸
        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_c * 16, base_c * 8)

        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2, output_padding=1)
        self.dec3 = DoubleConv(base_c * 8, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2, output_padding=1)
        self.dec2 = DoubleConv(base_c * 4, base_c * 2)

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2, output_padding=1)
        self.dec1 = DoubleConv(base_c * 2, base_c)

        # 最终输出层 - 添加额外的上采样以匹配目标尺寸
        self.final_up = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path with LSTM
        x1 = self.enc1(x)
        h1 = torch.zeros_like(x1)
        c1 = torch.zeros_like(x1)
        h1, c1 = self.lstm1(x1, h1, c1)
        
        x2 = self.pool1(h1)
        x2 = self.enc2(x2)
        h2 = torch.zeros_like(x2)
        c2 = torch.zeros_like(x2)
        h2, c2 = self.lstm2(x2, h2, c2)
        
        x3 = self.pool2(h2)
        x3 = self.enc3(x3)
        h3 = torch.zeros_like(x3)
        c3 = torch.zeros_like(x3)
        h3, c3 = self.lstm3(x3, h3, c3)
        
        x4 = self.pool3(h3)
        x4 = self.enc4(x4)
        h4 = torch.zeros_like(x4)
        c4 = torch.zeros_like(x4)
        h4, c4 = self.lstm4(x4, h4, c4)
        
        x5 = self.pool4(h4)
    
        # Bottleneck
        x5 = self.bottleneck(x5)
    
        # Decoder path
        x = self.up4(x5)
        # 确保h4与x尺寸匹配
        if x.shape[2:] != h4.shape[2:]:
            h4 = F.interpolate(h4, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, h4], dim=1)
        x = self.dec4(x)
    
        x = self.up3(x)
        # 确保h3与x尺寸匹配
        if x.shape[2:] != h3.shape[2:]:
            h3 = F.interpolate(h3, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, h3], dim=1)
        x = self.dec3(x)
    
        x = self.up2(x)
        # 确保h2与x尺寸匹配
        if x.shape[2:] != h2.shape[2:]:
            h2 = F.interpolate(h2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, h2], dim=1)
        x = self.dec2(x)
    
        x = self.up1(x)
        # 确保h1与x尺寸匹配
        if x.shape[2:] != h1.shape[2:]:
            h1 = F.interpolate(h1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, h1], dim=1)
        x = self.dec1(x)
        
        # 应用最终卷积并调整到目标尺寸
        x = self.final_conv(x)
        x = self.final_up(x)  # 确保输出尺寸与目标匹配
    
        return {"out": x}