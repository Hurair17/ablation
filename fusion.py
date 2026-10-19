# ultralytics/nn/modules/fusion.py

import torch
import torch.nn as nn
from conv import Conv  # Ultralytics Conv (Conv+BN+SiLU)


class GatedFusion2D(nn.Module):
    """
    Per-scale gated fusion of two feature maps:
      x = (x_cam, x_lidar)

    Channel align both to c_out, then compute a gate and blend.
    """

    def __init__(self, c_cam, c_lidar, c_out):
        super().__init__()
        self.cam_proj = Conv(c_cam, c_out, 1, 1)
        self.lidar_proj = Conv(c_lidar, c_out, 1, 1)

        # Learnable parameters for gating
        self.gamma = nn.Parameter(torch.zeros(1, c_out, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_out, 1, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cam, x_lidar):
        f_cam = self.cam_proj(x_cam)
        f_lidar = self.lidar_proj(x_lidar)

        # simple FiLM-like gating
        gate = self.sigmoid(self.gamma * f_cam + self.beta * f_lidar)
        fused = gate * f_cam + (1.0 - gate) * f_lidar
        return fused



class CrossAttnFusion2D(nn.Module):
    """
    Cross-attention fusion between camera and LiDAR feature maps.

    - camera->LiDAR: camera queries, LiDAR keys/values
    - LiDAR->camera: LiDAR queries, camera keys/values
    - apply dropout on each attention output
    - concat both attentions and project back to c_out

    Input:
      x_cam:   [B, C_cam, H, W]
      x_lidar: [B, C_lidar, H, W]
    Output:
      fused:   [B, c_out, H, W]
    """

    def __init__(self, c_cam, c_lidar, c_out, num_heads=4, dropout=0.1):
        super().__init__()

        self.c_out = c_out

        # Project camera and LiDAR to common d_model = c_out
        self.q_cam = Conv(c_cam, c_out, k=1, s=1)
        self.k_cam = Conv(c_cam, c_out, k=1, s=1)
        self.v_cam = Conv(c_cam, c_out, k=1, s=1)

        self.q_lidar = Conv(c_lidar, c_out, k=1, s=1)
        self.k_lidar = Conv(c_lidar, c_out, k=1, s=1)
        self.v_lidar = Conv(c_lidar, c_out, k=1, s=1)

        # Shared multihead attention for simplicity
        self.mha = nn.MultiheadAttention(
            embed_dim=c_out,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # we'll use (L, B, C)
        )

        self.dropout = nn.Dropout(dropout)

        # After getting cam->lidar and lidar->cam attentions (2 * c_out),
        # fuse them back to c_out
        self.out_proj = Conv(2 * c_out, c_out, k=1, s=1)

    def _attend(self, q, k, v):
        """
        q, k, v: [B, C, H, W]
        Returns: [B, C, H, W] after multihead attention.
        """
        B, C, H, W = q.shape
        L = H * W

        # [B, C, H, W] -> [B, C, L] -> [L, B, C]
        def flatten(x):
            return x.view(B, C, L).permute(2, 0, 1)

        q_f = flatten(q)
        k_f = flatten(k)
        v_f = flatten(v)

        attn_out, _ = self.mha(q_f, k_f, v_f)  # [L, B, C]
        # back to [B, C, H, W]
        attn_out = attn_out.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return attn_out

    def forward(self, x_cam, x_lidar):
        # project to q/k/v
        q_cam = self.q_cam(x_cam)
        k_cam = self.k_cam(x_cam)
        v_cam = self.v_cam(x_cam)

        q_lidar = self.q_lidar(x_lidar)
        k_lidar = self.k_lidar(x_lidar)
        v_lidar = self.v_lidar(x_lidar)

        # camera -> LiDAR attention (camera queries LiDAR)
        cam_to_lidar = self._attend(q_cam, k_lidar, v_lidar)  # [B, c_out, H, W]
        cam_to_lidar = self.dropout(cam_to_lidar)

        # LiDAR -> camera attention (LiDAR queries camera)
        lidar_to_cam = self._attend(q_lidar, k_cam, v_cam)    # [B, c_out, H, W]
        lidar_to_cam = self.dropout(lidar_to_cam)

        # concat the two attention outputs and project back
        fused = torch.cat([cam_to_lidar, lidar_to_cam], dim=1)  # [B, 2*c_out, H, W]
        fused = self.out_proj(fused)                            # [B, c_out, H, W]
        return fused
