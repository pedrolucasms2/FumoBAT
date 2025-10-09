# src/models/optimal_model.py
import torch
import torch.nn as nn
from .blocks import ConvBNAct, RFCBAMDownsample, C2f_PIG, CAA, DySample, FRM, ObjectRelationModule

class DecoupledDetectHead(nn.Module):
    """
    An improved, decoupled detection head.
    
    This head separates the classification and regression tasks into different
    branches, which can lead to improved performance by resolving the conflict
    between these two competing tasks.
    """
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        
        # A common stem to process features before they are sent to the branches.
        self.stem = ConvBNAct(in_channels, in_channels, 1, 1)
        
        # --- Classification Branch ---
        # A small stack of convolutional layers dedicated to learning classification features.
        self.cls_convs = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3, 1),
            ConvBNAct(in_channels, in_channels, 3, 1),
        )
        # Final predictor for class scores.
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        
        # --- Regression Branch ---
        # A separate stack for learning bounding box regression features.
        self.reg_convs = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3, 1),
            ConvBNAct(in_channels, in_channels, 3, 1)
        )
        # Predictor for bounding box coordinates.
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        # Predictor for the objectness score (whether an object is present).
        self.obj_pred = nn.Conv2d(in_channels, num_anchors * 1, 1)

    def forward(self, x):
        # 1. Pass input through the common stem.
        x_stem = self.stem(x)
        
        # 2. Process features in parallel through the classification and regression branches.
        cls_feat = self.cls_convs(x_stem)
        reg_feat = self.reg_convs(x_stem)
        
        # 3. Get the final predictions from each branch.
        cls_output = self.cls_pred(cls_feat)
        reg_output = self.reg_pred(reg_feat)
        obj_output = self.obj_pred(reg_feat)
        
        # 4. Concatenate the outputs in the standard YOLO order [reg, obj, cls].
        # The loss function expects this specific channel ordering.
        return torch.cat([reg_output, obj_output, cls_output], dim=1)


class BSSIFPN(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # Normalize all channels to c3 (256) for simplicity
        self.b1_ref = ConvBNAct(c1, c3, 1, 1)  # 64->256
        self.b2_ref = ConvBNAct(c2, c3, 1, 1)  # 128->256
        self.p4_lat = ConvBNAct(c4, c3, 1, 1)  # 384->256
        self.p3_lat = ConvBNAct(c3, c3, 1, 1)  # 256->256
        self.p2_lat = ConvBNAct(c3, c3, 1, 1)  # 256->256 (after B2 ref)
        
        # Dynamic upsampling
        self.up4 = DySample(c3, scale=2)
        self.up3 = DySample(c3, scale=2)
        
        # FRMs with compatible channels (all c3 = 256)
        self.frm3 = FRM(c3, c3)  # up=256, skip=256
        self.frm2 = FRM(c3, c3)  # up=256, skip=256
        
        # Inject B1 into P2
        self.b1_to_p2 = ConvBNAct(c3, c3, 3, 2)  # 256->256
        
        # Additional bottom-up path
        self.down_p2 = ConvBNAct(c3, c3, 3, 2)  # 256->256
        self.frm_bu_p3 = FRM(c3, c3)  # up=256, skip=256
        self.down_p3 = ConvBNAct(c3, c3, 3, 2)  # 256->256
        self.frm_bu_p4 = FRM(c3, c3)  # up=256, skip=256
        
        # Final outputs - adjust to the heads' channel requirements
        self.out2 = ConvBNAct(c3, c2, 3, 1)  # 256->128 for P2
        self.out3 = ConvBNAct(c3, c3, 3, 1)  # 256->256 for P3
        self.out4 = ConvBNAct(c3, c3, 3, 1)  # 256->256 for P4

    def forward(self, B1, B2, B3, B4):
        # Normalize everything to c3 (256 channels)
        B1r = self.b1_ref(B1)  # 64->256
        B2r = self.b2_ref(B2)  # 128->256
        B3r = self.p3_lat(B3)  # 256->256
        B4r = self.p4_lat(B4)  # 384->256
        
        # Top-down path
        P4 = B4r  # 256 channels
        P3 = self.frm3(self.up4(P4), B3r)  # 256+256
        P2 = self.frm2(self.up3(P3), self.p2_lat(B2r))  # 256+256
        
        # Inject B1
        P2 = self.frm2(P2, self.b1_to_p2(B1r))  # 256+256
        
        # Additional bottom-up path
        D3 = self.down_p2(P2)  # 256
        P3 = self.frm_bu_p3(P3, D3)  # 256+256
        
        D4 = self.down_p3(P3)  # 256
        P4 = self.frm_bu_p4(P4, D4)  # 256+256
        
        # Adjust output channels as needed
        return self.out2(P2), self.out3(P3), self.out4(P4)

class SmallObjectYOLO(nn.Module):
    def __init__(self, nc=80, ch=(64, 128, 256, 384)):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.b1 = C2f_PIG(c1, c1, mode='heavy')
        self.down12, self.b2 = RFCBAMDownsample(c1, c2), C2f_PIG(c2, c2, mode='heavy')
        self.down23, self.b3 = RFCBAMDownsample(c2, c3), C2f_PIG(c3, c3, mode='light')
        self.down34, self.b4 = RFCBAMDownsample(c3, c4), C2f_PIG(c4, c4, mode='light')
        self.caa = CAA(c4, k=9)
        self.neck = BSSIFPN(c1, c2, c3, c4)

        self.relation_p2 = ObjectRelationModule(c_in=c2) # 128 channels
        self.relation_p3 = ObjectRelationModule(c_in=c3) # 256 channels
        self.relation_p4 = ObjectRelationModule(c_in=c3) # 256 channels (neck output for P4 is c3)

        # --- Use the new DecoupledDetectHead for potentially better results ---
        self.head_p2 = DecoupledDetectHead(c2, nc)  # 128 channels
        self.head_p3 = DecoupledDetectHead(c3, nc)  # 256 channels  
        self.head_p4 = DecoupledDetectHead(c3, nc)  # 256 channels
        
        self.strides = torch.tensor([8., 16., 32.])

    def forward(self, x):
        x = self.stem(x)          # 1/2
        B1 = self.b1(x)           # 1/2
        B2 = self.b2(self.down12(B1))  # 1/4
        B3 = self.b3(self.down23(B2))  # 1/8
        B4 = self.b4(self.down34(B3))  # 1/16
        B4 = self.caa(B4)
        P2, P3, P4 = self.neck(B1, B2, B3, B4)

        P2_rel = self.relation_p2(P2)
        P3_rel = self.relation_p3(P3)
        P4_rel = self.relation_p4(P4)
        
        # Pass the processed features through the detection heads.
        out_p2 = self.head_p2(P2_rel)
        out_p3 = self.head_p3(P3_rel)
        out_p4 = self.head_p4(P4_rel)

        return [out_p2, out_p3, out_p4], self.strides.to(x.device)
