# ultralytics/nn/modules/backbone_yolo11.py

import torch.nn as nn
from conv import Conv
from block import C3k2
# from attention import C2PSA
from block import C2PSA

from block import SPPF
# Yolov11m
class YOLO11Backbone(nn.Module):
    """
    YOLOv11m backbone - medium variant with 1.5x channels.
    Returns (P3, P4, P5).
    """
    def __init__(self, ch_in=3):
        super().__init__()

        # YOLOv11m uses width_multiple=0.75, so scale all channels by 1.5x
        self.layer0 = Conv(ch_in, 96, 3, 2)      # 64*1.5=96
        self.layer1 = Conv(96, 192, 3, 2)        # 128*1.5=192
        self.layer2 = C3k2(192, 384, n=2, shortcut=False, e=0.5)  # 256*1.5=384
        self.layer3 = Conv(384, 384, 3, 2)
        self.layer4 = C3k2(384, 768, n=2, shortcut=False, e=0.5)  # 512*1.5=768
        self.layer5 = Conv(768, 768, 3, 2)
        self.layer6 = C3k2(768, 768, n=2, shortcut=True, e=0.5)
        self.layer7 = Conv(768, 1536, 3, 2)      # 1024*1.5=1536
        self.layer8 = C3k2(1536, 1536, n=2, shortcut=True, e=0.5)
        self.layer9 = SPPF(1536, 1536, k=5)
        self.layer10 = C2PSA(1536, 1536, n=1, e=0.5)  # Use e=0.5 for proper attention dims
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_p3 = self.layer3(x)        # 3-P3/8
        x_p3 = self.layer4(x_p3)
        x_p4 = self.layer5(x_p3)     # 5-P4/16
        x_p4 = self.layer6(x_p4)
        x_p5 = self.layer7(x_p4)     # 7-P5/32
        x_p5 = self.layer8(x_p5)
        x_p5 = self.layer9(x_p5)
        x_p5 = self.layer10(x_p5)
        return x_p3, x_p4, x_p5


# Yolov11n (for LiDAR backbone)
class YOLO11nBackbone(nn.Module):
    """
    YOLOv11n backbone - nano variant for LiDAR.
    Returns (P3, P4, P5) with 512, 512, 1024 channels.
    """
    def __init__(self, ch_in=3):
        super().__init__()

        self.layer0 = Conv(ch_in, 64, 3, 2)      # 0-P1/2
        self.layer1 = Conv(64, 128, 3, 2)        # 1-P2/4
        self.layer2 = C3k2(128, 256, n=2, shortcut=False, e=0.25)
        self.layer3 = Conv(256, 256, 3, 2)       # 3-P3/8
        self.layer4 = C3k2(256, 512, n=2, shortcut=False, e=0.25)
        self.layer5 = Conv(512, 512, 3, 2)       # 5-P4/16
        self.layer6 = C3k2(512, 512, n=2, shortcut=True)  # P4
        self.layer7 = Conv(512, 1024, 3, 2)      # 7-P5/32
        self.layer8 = C3k2(1024, 1024, n=2, shortcut=True)
        self.layer9 = SPPF(1024, 1024, k=5)      # 9
        self.layer10 = C2PSA(1024, 1024)         # 10 = P5

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_p3 = self.layer3(x)        # 3-P3/8
        x_p3 = self.layer4(x_p3)
        x_p4 = self.layer5(x_p3)     # 5-P4/16
        x_p4 = self.layer6(x_p4)
        x_p5 = self.layer7(x_p4)     # 7-P5/32
        x_p5 = self.layer8(x_p5)
        x_p5 = self.layer9(x_p5)
        x_p5 = self.layer10(x_p5)
        return x_p3, x_p4, x_p5
