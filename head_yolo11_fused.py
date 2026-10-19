
import torch
import torch.nn as nn
from conv import Conv
from block import C3k2
from head import Detect  # or adjust import path to match your repo


# Yolov11m
class YOLO11HeadFused(nn.Module):
    """
    YOLOv11m neck + Detect, taking fused P3(768), P4(768), P5(1536).
    Medium variant with 1.5x channels.
    """
    def __init__(self, nc=80):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # YOLOv11m channels: P3=768, P4=768, P5=1536
        self.c3_p4 = C3k2(1536 + 768, 768, n=2, shortcut=False)  # Changed
        self.c3_p3 = C3k2(768 + 768, 384, n=2, shortcut=False)   # Changed

        self.down_p3 = Conv(384, 384, 3, 2)                      # Changed
        self.c3_n4 = C3k2(384 + 768, 768, n=2, shortcut=False)   # Changed

        self.down_p4 = Conv(768, 768, 3, 2)                      # Changed
        self.c3_n5 = C3k2(768 + 1536, 1536, n=2, shortcut=True)  # Changed

        # Detect with medium channels
        self.detect = Detect(nc=nc, ch=(384, 768, 1536))         # Changed


# Yolov11n
# class YOLO11HeadFused(nn.Module):
#     """
#     YOLO11 neck + Detect, taking fused P3(512), P4(512), P5(1024).
#     Mirrors the yolo11.yaml head and outputs [P3_small(256), P4(512), P5(1024)].
#     """

#     def __init__(self, nc=80):
#         super().__init__()

#         self.up = nn.Upsample(scale_factor=2, mode="nearest")

#         # from YAML:
#         # - cat(p5_up(1024), p4(512)) -> 1536 -> C3k2(..., 512)
#         self.c3_p4 = C3k2(1024 + 512, 512, n=2, shortcut=False)  # head layer 13

#         # - cat(p4_up(512), p3(512)) -> 1024 -> C3k2(..., 256)
#         self.c3_p3 = C3k2(512 + 512, 256, n=2, shortcut=False)   # head layer 16

#         # downsample small -> mid
#         self.down_p3 = Conv(256, 256, 3, 2)                      # head layer 17
#         self.c3_n4 = C3k2(256 + 512, 512, n=2, shortcut=False)   # head layer 19

#         # downsample mid -> large
#         self.down_p4 = Conv(512, 512, 3, 2)                      # head layer 20
#         self.c3_n5 = C3k2(512 + 1024, 1024, n=2, shortcut=True)  # head layer 22

#         # Detect on 3 scales: (P3 small, P4 medium, P5 large)
#         self.detect = Detect(nc=nc, ch=(256, 512, 1024))



    def forward(self, p3, p4, p5):
        # p3: [B, 512, H/8,  W/8]
        # p4: [B, 512, H/16, W/16]
        # p5: [B, 1024, H/32, W/32]

        # Top-down: P5 -> P4
        x11 = p5
        x12 = self.up(x11)                         # [B, 1024, H/16, W/16]
        x13 = self.c3_p4(torch.cat([x12, p4], 1))  # [B, 512, H/16, W/16]

        # Top-down: P4 -> P3
        x14 = self.up(x13)                         # [B, 512, H/8, W/8]
        x16 = self.c3_p3(torch.cat([x14, p3], 1))  # [B, 256, H/8, W/8]
        n3 = x16                                   # small

        # Bottom-up: P3 -> P4
        x18 = self.down_p3(n3)                     # [B, 256, H/16, W/16]
        x19 = self.c3_n4(torch.cat([x18, x13], 1)) # [B, 512, H/16, W/16]
        n4 = x19                                   # medium

        # Bottom-up: P4 -> P5
        x20 = self.down_p4(n4)                     # [B, 512, H/32, W/32]
        x22 = self.c3_n5(torch.cat([x20, p5], 1))  # [B, 1024, H/32, W/32]
        n5 = x22                                   # large

        return self.detect([n3, n4, n5])
