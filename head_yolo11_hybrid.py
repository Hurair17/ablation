
import torch
import torch.nn as nn
from conv import Conv
from block import C3k2
from head import Detect


class YOLO11HybridHeadFused(nn.Module):
    """
    Hybrid YOLO11 head for asymmetric fusion:
    - Camera: YOLOv11m (P3=768, P4=768, P5=1536)
    - LiDAR: YOLOv11n (P3=512, P4=512, P5=1024)

    After fusion, uses medium channels for the head.
    """
    def __init__(self, nc=80, cam_p3_ch=768, cam_p4_ch=768, cam_p5_ch=1536):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Assuming fusion outputs match camera channels (after fusion module)
        # Top-down path
        self.c3_p4 = C3k2(cam_p5_ch + cam_p4_ch, cam_p4_ch, n=2, shortcut=False)
        self.c3_p3 = C3k2(cam_p4_ch + cam_p3_ch, 384, n=2, shortcut=False)

        # Bottom-up path
        self.down_p3 = Conv(384, 384, 3, 2)
        self.c3_n4 = C3k2(384 + cam_p4_ch, cam_p4_ch, n=2, shortcut=False)

        self.down_p4 = Conv(cam_p4_ch, cam_p4_ch, 3, 2)
        self.c3_n5 = C3k2(cam_p4_ch + cam_p5_ch, cam_p5_ch, n=2, shortcut=True)

        # Detect layer
        self.detect = Detect(nc=nc, ch=(384, cam_p4_ch, cam_p5_ch))

    def forward(self, p3, p4, p5):
        """
        Args:
            p3: Fused P3 features [B, cam_p3_ch, H/8, W/8]
            p4: Fused P4 features [B, cam_p4_ch, H/16, W/16]
            p5: Fused P5 features [B, cam_p5_ch, H/32, W/32]
        """
        # Top-down: P5 -> P4
        x11 = p5
        x12 = self.up(x11)
        x13 = self.c3_p4(torch.cat([x12, p4], 1))

        # Top-down: P4 -> P3
        x14 = self.up(x13)
        x16 = self.c3_p3(torch.cat([x14, p3], 1))
        n3 = x16  # small

        # Bottom-up: P3 -> P4
        x18 = self.down_p3(n3)
        x19 = self.c3_n4(torch.cat([x18, x13], 1))
        n4 = x19  # medium

        # Bottom-up: P4 -> P5
        x20 = self.down_p4(n4)
        x22 = self.c3_n5(torch.cat([x20, p5], 1))
        n5 = x22  # large

        return self.detect([n3, n4, n5])
