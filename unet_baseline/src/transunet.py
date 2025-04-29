import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

class TransUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_c=32, patch_size=16, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        # Encoder部分
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True)
        )
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_c, base_c*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c*2, base_c*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*2),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_c*2, base_c*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c*4, base_c*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*4),
            nn.ReLU(inplace=True)
        )
        
        # Transformer部分
        self.patch_embed = nn.Conv2d(base_c*4, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (100//(patch_size*4))**2, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_expand = nn.Linear(embed_dim, patch_size * patch_size * base_c*4)
        
        # Decoder部分
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2),
            nn.Conv2d(base_c*2, base_c*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c*2, base_c*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*2),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True)
        )
        
        self.up_final = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(base_c, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Transformer
        x = self.patch_embed(x3)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # 恢复空间维度
        B, N, C = x.shape
        H = W = int(N**0.5)
        x = self.patch_expand(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                     h=H, w=W, p1=16, p2=16)
        
        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.up_final(x)  # 添加最终的上采样层
        x = self.outc(x)
        
        return {"out": x}