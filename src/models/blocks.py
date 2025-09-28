
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

class ObjectRelationModule(nn.Module):
    def __init__(self, c_in, n_relations=16, d_k=64, d_g=64):
        super().__init__()
        self.n_relations = n_relations
        self.d_k = d_k
        self.d_g = d_g
    
        self.W_k = nn.Linear(c_in, d_k)
        self.W_q = nn.Linear(c_in, d_k)
        self.W_v = nn.Linear(c_in, c_in)
    
        self.geo_embedding = nn.Sequential(
            nn.Linear(4, d_g),
            nn.ReLU(),
            nn.Linear(d_g, d_g),
            nn.ReLU()
        )
            
        self.W_g = nn.Linear(d_g, n_relations)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W 
            
        MAX_FEATURES = 8192 
         
        features = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)        
    
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        bboxes = torch.stack([xx.float(), yy.float(), torch.ones_like(xx), torch.ones_like(yy)], dim=-1).view(N, 4)        
    
        if N > MAX_FEATURES:
            print(f"[WARNING] Too many features ({N}), sampling {MAX_FEATURES} features")            
        
            step = int(np.sqrt(N / MAX_FEATURES)) 
            indices = []
            for i in range(0, H, step):
                for j in range(0, W, step):
                    if len(indices) < MAX_FEATURES:
                        indices.append(i * W + j)
            
            indices = torch.tensor(indices, device=x.device)[:MAX_FEATURES]
                    
            features = features[:, indices, :] 
            bboxes = bboxes[indices] 
            N = len(indices)
            
            print(f"[INFO] Reduced to {N} features")
            
        diff = bboxes.unsqueeze(0) - bboxes.unsqueeze(1)
            
        geo_features = torch.log(torch.abs(diff) + 1e-8)
            
        embedded_geo = self.geo_embedding(geo_features)
            
        w_g = F.relu(self.W_g(embedded_geo))

        q = self.W_q(features) 
        k = self.W_k(features) 
            
        w_a = torch.bmm(q, k.transpose(1, 2)) / (self.d_k ** 0.5)
        w_a = w_a.softmax(dim=2)
    
        w_g_mean = w_g.mean(dim=2).unsqueeze(0)
            
        w_mn = w_a * w_g_mean
    
        v = self.W_v(features)
            
        relation_features = torch.bmm(w_mn, v)
            
        augmented_features = features + relation_features
            
        if N < H * W: 

            print(f"[WARNING] Returning original feature map due to sampling")
            return x
        else:
        
            return augmented_features.view(B, H, W, C).permute(0, 3, 1, 2)