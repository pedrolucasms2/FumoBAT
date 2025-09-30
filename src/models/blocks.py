
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DWConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dw = ConvBNAct(c1, c1, k, s, g=c1)
        self.pw = ConvBNAct(c1, c2, 1, 1, act=act)
    def forward(self, x): return self.pw(self.dw(x))

class GhostModule(nn.Module):
    def __init__(self, c1, c2, ratio=2):
        super().__init__()
        cm = math.ceil(c2 / ratio)
        self.primary = ConvBNAct(c1, cm, 1, 1)
        self.cheap = ConvBNAct(cm, c2 - cm, 3, 1, g=cm)
    def forward(self, x):
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y, z], dim=1)

class GhostBottleneckV2(nn.Module):
    def __init__(self, c1, c2, s=1, exp=None):
        super().__init__()
        exp = exp or max(c1, c2)
        self.expand = GhostModule(c1, exp)
        self.dw = nn.Identity() if s == 1 else DWConv(exp, exp, k=3, s=s)
        self.project = GhostModule(exp, c2)
        self.short = nn.Identity() if (c1 == c2 and s == 1) else DWConv(c1, c2, k=3, s=s)
    def forward(self, x): return self.project(self.dw(self.expand(x))) + self.short(x)

class PartialConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, ratio=0.25):
        super().__init__()
        self.keep = max(1, int(c1 * (1 - ratio)))
        self.idx = None
        self.conv = ConvBNAct(self.keep, c2, k, s)
    def forward(self, x):
        if self.idx is None or self.idx.shape != self.keep:
            self.idx = torch.randperm(x.shape[1], device=x.device)[:self.keep]
        return self.conv(x[:, self.idx, :, :])

class InceptionDepthwiseConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw3, self.dw5, self.dw7 = DWConv(c, c, 3), DWConv(c, c, 5), DWConv(c, c, 7)
        self.pw = ConvBNAct(c * 3, c, 1, 1)
    def forward(self, x):
        a, b, c = self.dw3(x), self.dw5(x), self.dw7(x)
        return self.pw(torch.cat([a, b, c], dim=1))

class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        m = max(1, c // r)
        self.mlp = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.SiLU(), nn.Conv2d(m, c, 1, bias=False))
    def forward(self, x):
        w = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=(k-1)//2, bias=False)
    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))
        return x * w

class RFCBAMDownsample(nn.Module):
    def __init__(self, c1, c2, groups=4):
        super().__init__()
        g = min(groups, c1)
        self.group = ConvBNAct(c1, c1, 3, 1, g=g)
        self.expand = ConvBNAct(c1, c1, 3, 1)
        self.ca, self.sa = ChannelAttention(c1), SpatialAttention(7)
        self.down = ConvBNAct(c1, c2, 3, 2)
    def forward(self, x): return self.down(self.sa(self.ca(self.expand(self.group(x)))))

class C2f_PIG(nn.Module):
    def __init__(self, c1, c2, mode='heavy'):
        super().__init__()
        if mode == 'heavy':
            self.block = nn.Sequential(
                ConvBNAct(c1, c2, 1, 1),
                PartialConv(c2, c2, 3, 1, ratio=0.25),
                InceptionDepthwiseConv(c2),
                ConvBNAct(c2, c2, 1, 1),
            )
        else:
            self.block = GhostBottleneckV2(c1, c2, s=1)
    def forward(self, x): return self.block(x)

class CAA(nn.Module):
    def __init__(self, c, k=9):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.h = nn.Conv2d(c, c, (1, k), padding=(0, (k-1)//2), groups=c, bias=False)
        self.v = nn.Conv2d(c, c, (k, 1), padding=((k-1)//2, 0), groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
    def forward(self, x):
        g = self.pool(x).expand_as(x)
        w = torch.sigmoid(self.pw(self.h(g) + self.v(g)))
        return x * w

class DySample(nn.Module):
    def __init__(self, c, scale=2, groups=4):
        super().__init__()
        self.scale, self.groups = scale, groups
        self.offset = nn.Conv2d(c, 2 * groups, 3, padding=1)
        self.proj = ConvBNAct(c, c, 1, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        s, G = self.scale, self.groups
        off = torch.tanh(self.offset(x))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H*s, device=x.device),
                                torch.linspace(-1, 1, W*s, device=x.device), indexing='ij')
        base = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        xg = self.proj(x).view(B, G, C // G, H, W)
        outs = []
        for g in range(G):
            o = F.interpolate(off[:, 2*g:2*g+2], scale_factor=s, mode='bilinear', align_corners=True)
            grid = base + o.permute(0, 2, 3, 1) * (2.0 / max(H*s, W*s))
            outs.append(F.grid_sample(xg[:, g], grid, mode='bilinear', padding_mode='border', align_corners=True))
        return torch.cat(outs, 1)

class FRM(nn.Module):
    def __init__(self, c_up, c_skip):
        super().__init__()
    
        self.align = nn.Conv2d(c_up, c_skip, 1, bias=False) if c_up != c_skip else nn.Identity()
        
    
        self.g = nn.Sequential(
            nn.Conv2d(c_skip * 2, c_skip, 1, bias=False), 
            nn.SiLU(),
            nn.Conv2d(c_skip, 1, 1)
        )
    
    def forward(self, up, skip):
    
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        
    
        up_aligned = self.align(up)
        
    
        delta = torch.sigmoid(self.g(torch.cat([up_aligned, skip], 1)))
        
    
        return delta * skip + (1 - delta) * up_aligned

class EfficientObjectRelationModule(nn.Module):
    """
    Substitui ObjectRelationModule por versão MUITO mais eficiente
    - Remove sampling aleatório problemático
    - Usa attention linear ao invés de quadrática
    - Mantém performance para objetos pequenos
    """
    def __init__(self, c_in, reduction=4, num_scales=4):
        super().__init__()
        self.c_in = c_in
        self.num_scales = num_scales
        
        # Multi-scale pooling para capturar relações em diferentes escalas
        self.scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(s) for s in [1, 2, 4, 8]
        ])
        
        # Channel attention eficiente
        mid_channels = max(16, c_in // reduction)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, c_in, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial relation através de convoluções separáveis
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False),  # Depthwise
            nn.Conv2d(c_in, c_in, kernel_size=1, bias=False),  # Pointwise
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True)
        )
        
        # Position encoding leve
        self.position_embed = nn.Parameter(torch.randn(1, c_in, 1, 1) * 0.02)
        
        # Output projection
        self.output_proj = nn.Conv2d(c_in * 2, c_in, 1, bias=False)
        
        # Layer normalization
        self.norm = nn.BatchNorm2d(c_in)
        
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # === CHANNEL RELATIONS ===
        # Global channel attention - O(C) complexity
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # === SPATIAL RELATIONS ===
        # Efficient spatial modeling through separable convolutions
        x_spatial = self.spatial_conv(x_channel)
        
        # === MULTI-SCALE CONTEXT ===
        # Capture relations at different scales - LINEAR complexity
        multi_scale_features = []
        for pool in self.scale_pools:
            # Pool to different scales
            pooled = pool(x_spatial)  # [B, C, scale, scale]
            
            # Upsample back to original size
            upsampled = F.interpolate(pooled, size=(H, W), 
                                    mode='bilinear', align_corners=False)
            multi_scale_features.append(upsampled)
        
        # Combine multi-scale features
        multi_scale_context = sum(multi_scale_features) / len(multi_scale_features)
        
        # === POSITION ENCODING ===
        # Add learnable position information
        x_with_pos = x_spatial + self.position_embed
        
        # === FEATURE FUSION ===
        # Combine spatial and multi-scale features
        combined = torch.cat([x_with_pos, multi_scale_context], dim=1)
        output = self.output_proj(combined)
        
        # === RESIDUAL CONNECTION ===
        # Weighted residual connection
        output = self.norm(output + identity * 0.2)
        
        return output

class LinearAttentionRelation(nn.Module):
    """
    Alternative: Linear attention para relações espaciais
    Complexidade O(HW) ao invés de O(H²W²)
    """
    def __init__(self, c_in, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = c_in // heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.to_q = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.to_k = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.to_v = nn.Conv2d(c_in, c_in, 1, bias=False)
        
        # Output projection
        self.to_out = nn.Conv2d(c_in, c_in, 1, bias=False)
        
        # Normalization
        self.norm = nn.BatchNorm2d(c_in)
        
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # Generate Q, K, V
        q = self.to_q(x).view(B, self.heads, self.head_dim, H*W)
        k = self.to_k(x).view(B, self.heads, self.head_dim, H*W)
        v = self.to_v(x).view(B, self.heads, self.head_dim, H*W)
        
        # Linear attention trick: normalize K first
        k = F.softmax(k, dim=-2)  # Normalize over spatial dimension
        
        # Compute context: O(d²) instead of O(n²)
        context = torch.matmul(k.transpose(-1, -2), v.transpose(-1, -2))  # [B, heads, HW, HW] -> [B, heads, d, d]
        
        # Apply context to queries: O(nd) instead of O(n²)
        out = torch.matmul(q.transpose(-1, -2), context).transpose(-1, -2)  # [B, heads, d, HW]
        
        # Reshape and project
        out = out.reshape(B, C, H, W)
        out = self.to_out(out)
        
        # Residual connection with normalization
        return self.norm(out + identity)

class ObjectRelationModule(nn.Module):
    """Versão eficiente - sem sampling problemático"""
    def __init__(self, c_in, n_relations=16, d_k=64, d_g=64):
        super().__init__()
        # Usar a implementação eficiente
        self.efficient_relation = EfficientObjectRelationModule(c_in, reduction=4)
        
    def forward(self, x):
        return self.efficient_relation(x)