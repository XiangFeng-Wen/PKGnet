import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                          attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class TSESNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_c=32, bilinear=True,
                 embed_dim=256, depth=4, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1):
        super(TSESNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Encoder path with attention modules
        self.in_conv = DoubleConv(in_channels, base_c)
        self.ca1 = ChannelAttention(base_c)
        self.sa1 = SpatialAttention()
        
        self.down1 = Down(base_c, base_c * 2)
        self.ca2 = ChannelAttention(base_c * 2)
        self.sa2 = SpatialAttention()
        
        self.down2 = Down(base_c * 2, base_c * 4)
        self.ca3 = ChannelAttention(base_c * 4)
        self.sa3 = SpatialAttention()
        
        self.down3 = Down(base_c * 4, base_c * 8)
        self.ca4 = ChannelAttention(base_c * 8)
        self.sa4 = SpatialAttention()
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        
        # Transformer blocks at the bottleneck
        self.patch_size = 2  # Small patch size for feature maps
        self.transformer_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(base_c * 16 // factor, embed_dim, 
                                    kernel_size=self.patch_size, stride=self.patch_size)
        
        # Position embedding - will be resized dynamically based on input
        self.pos_embed = nn.Parameter(torch.zeros(1, 36, embed_dim))  # Initial size, will be interpolated
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        
        # Projection back to spatial domain
        self.proj_back = nn.ConvTranspose2d(embed_dim, base_c * 16 // factor, 
                                           kernel_size=self.patch_size, stride=self.patch_size)
        
        # Decoder path
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        # Encoder path with attention
        x1 = self.in_conv(x)
        x1 = x1 * self.ca1(x1)  # Apply channel attention
        x1 = x1 * self.sa1(x1)  # Apply spatial attention
        
        x2 = self.down1(x1)
        x2 = x2 * self.ca2(x2)
        x2 = x2 * self.sa2(x2)
        
        x3 = self.down2(x2)
        x3 = x3 * self.ca3(x3)
        x3 = x3 * self.sa3(x3)
        
        x4 = self.down3(x3)
        x4 = x4 * self.ca4(x4)
        x4 = x4 * self.sa4(x4)
        
        x5 = self.down4(x4)
        
        # Transformer blocks
        # Convert to patches
        B, C, H, W = x5.shape
        x_patch = self.patch_embed(x5)  # [B, embed_dim, H/patch_size, W/patch_size]
        x_patch = x_patch.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add position embeddings - dynamically resize to match feature map size
        num_patches = x_patch.shape[1]
        if num_patches != self.pos_embed.shape[1]:
            # Interpolate position embeddings to match current patch count
            pos_embed = self.pos_embed
            pos_embed_reshaped = pos_embed.reshape(1, int(np.sqrt(self.pos_embed.shape[1])), 
                                                int(np.sqrt(self.pos_embed.shape[1])), -1).permute(0, 3, 1, 2)
            h = w = int(np.sqrt(num_patches))
            pos_embed_interpolated = F.interpolate(pos_embed_reshaped, size=(h, w), mode='bilinear')
            pos_embed_interpolated = pos_embed_interpolated.permute(0, 2, 3, 1).reshape(1, num_patches, -1)
            x_patch = x_patch + pos_embed_interpolated
        else:
            x_patch = x_patch + self.pos_embed
        x_patch = self.pos_drop(x_patch)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x_patch = block(x_patch)
        
        # Project back to spatial domain
        x_spatial = x_patch.transpose(1, 2).reshape(B, self.transformer_dim, H // self.patch_size, W // self.patch_size)
        x5 = self.proj_back(x_spatial)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}