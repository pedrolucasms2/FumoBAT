# src/models/model.py
import torch
import torch.nn as nn
from .blocks import ConvBNAct, RFCBAMDownsample, C2f_PIG, CAA, DySample, FRM

class DetectHead(nn.Module):
    def __init__(self, c, nc):
        super().__init__()
        m = max(64, c // 2)
        self.stem = ConvBNAct(c, m, 1, 1)
        self.cls = nn.Sequential(ConvBNAct(m, m, 3, 1), nn.Conv2d(m, nc, 1))
        self.reg = nn.Sequential(ConvBNAct(m, m, 3, 1), nn.Conv2d(m, 4, 1))
        self.obj = nn.Sequential(ConvBNAct(m, m, 3, 1), nn.Conv2d(m, 1, 1))
    def forward(self, x):
        x = self.stem(x)
        return torch.cat([self.reg(x), self.obj(x), self.cls(x)], dim=1)

class BSSIFPN(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_ref = ConvBNAct(c1, c2, 1, 1)
        self.b2_ref = ConvBNAct(c2, c2, 1, 1)
        self.p4_lat = ConvBNAct(c4, c3, 1, 1)
        self.p3_lat = ConvBNAct(c3, c3, 1, 1)
        self.p2_lat = ConvBNAct(c2, c2, 1, 1)
        self.up4 = DySample(c3, scale=2)
        self.up3 = DySample(c3, scale=2)
        self.frm3 = FRM(c3)
        self.frm2 = FRM(c2)
        self.b1_to_p2 = ConvBNAct(c2, c2, 3, 2)
        self.down_p2 = ConvBNAct(c2, c3, 3, 2)
        self.frm_bu_p3 = FRM(c3)
        self.down_p3 = ConvBNAct(c3, c3, 3, 2)
        self.frm_bu_p4 = FRM(c3)
        self.out2, self.out3, self.out4 = ConvBNAct(c2, c2, 3, 1), ConvBNAct(c3, c3, 3, 1), ConvBNAct(c3, c3, 3, 1)

    def forward(self, B1, B2, B3, B4):
        B1r, B2r = self.b1_ref(B1), self.b2_ref(B2)
        P4 = self.p4_lat(B4)
        P3 = self.frm3(self.up4(P4), self.p3_lat(B3))
        P2 = self.frm2(self.up3(P3), self.p2_lat(B2r))
        P2 = self.frm2(P2, self.b1_to_p2(B1r))
        D3 = self.down_p2(P2); P3 = self.frm_bu_p3(P3, D3)
        D4 = self.down_p3(P3); P4 = self.frm_bu_p4(P4, D4)
        return self.out2(P2), self.out3(P3), self.out4(P4)

class SmallObjectYOLO(nn.Module):
    def __init__(self, nc=1, ch=(64, 128, 256, 384)):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.b1 = C2f_PIG(c1, c1, mode='heavy')
        self.down12, self.b2 = RFCBAMDownsample(c1, c2), C2f_PIG(c2, c2, mode='heavy')
        self.down23, self.b3 = RFCBAMDownsample(c2, c3), C2f_PIG(c3, c3, mode='light')
        self.down34, self.b4 = RFCBAMDownsample(c3, c4), C2f_PIG(c4, c4, mode='light')
        self.caa = CAA(c4, k=9)
        self.neck = BSSIFPN(c1, c2, c3, c4)
        self.head_p2, self.head_p3, self.head_p4 = DetectHead(c2, nc), DetectHead(c3, nc), DetectHead(c3, nc)
        self.strides = [4, 8, 16]

    def forward(self, x):
        x = self.stem(x)          # 1/2
        B1 = self.b1(x)           # 1/2
        B2 = self.b2(self.down12(B1))  # 1/4
        B3 = self.b3(self.down23(B2))  # 1/8
        B4 = self.b4(self.down34(B3))  # 1/16
        B4 = self.caa(B4)
        P2, P3, P4 = self.neck(B1, B2, B3, B4)
        return [self.head_p2(P2), self.head_p3(P3), self.head_p4(P4)], self.strides

if __name__ == "__main__":
    m = SmallObjectYOLO(nc=2)
    y, s = m(torch.randn(1, 3, 640, 640))
    print([t.shape for t in y], s)
