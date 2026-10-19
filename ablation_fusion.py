# ablation_fusion.py
# -------------------
# Ablation fusion modules for the MBiCMF ablation study.
# Each class isolates one component of CrossAttnFusion2D.
#
# A1  Full MBiCMF                    CrossAttnFusion2D  (in multimodal_yolo11.py)
# A2  Unidirectional Cam→LiDAR only  CrossAttnUniCam
# A3  Unidirectional LiDAR→Cam only  CrossAttnUniLidar
# A4  Single-scale P5 only           CameraPassthrough  (used for P3/P4)
# A5  No residual connections        CrossAttnNoResidual
# A6  No channel bottleneck          CrossAttnNoBottleneck
# A7  No FFN block                   CrossAttnNoFFN
# A8  No spatial dropout             CrossAttnNoDropout
# A9  Simple concatenation           ConcatFusion

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _proj_up(in_ch, out_ch):
    num_groups = 32 if out_ch % 32 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.GroupNorm(num_groups, out_ch),
        nn.SiLU(),
    )


def _ffn(dim, dropout):
    return nn.Sequential(
        nn.Linear(dim, dim * 4), nn.SiLU(), nn.Dropout(dropout),
        nn.Linear(dim * 4, dim), nn.Dropout(dropout),
    )


# ─────────────────────────────────────────────────────────────────────────────
# A2: Unidirectional — Camera queries LiDAR only (Branch 1 only)
# Ablates: bidirectionality (removes LiDAR→Camera branch entirely)
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttnUniCam(nn.Module):
    """A2: Q=cam, K=V=lidar — camera attends to LiDAR; no reverse branch."""
    def __init__(self, in_channels_cam, in_channels_lidar, out_channels,
                 num_heads=8, dropout=0.1, bottleneck_dim=256):
        super().__init__()
        D = bottleneck_dim
        self.proj_cam    = nn.Conv2d(in_channels_cam,   D, 1, bias=False)
        self.proj_lidar  = nn.Conv2d(in_channels_lidar, D, 1, bias=False)
        self.norm1_cam   = nn.LayerNorm(D)
        self.norm1_lidar = nn.LayerNorm(D)
        self.attn        = nn.MultiheadAttention(D, num_heads, dropout=dropout)
        self.norm2       = nn.LayerNorm(D)
        self.ffn         = _ffn(D, dropout)
        self.dropout_pre = nn.Dropout2d(p=min(dropout, 0.1))
        self.proj_up     = _proj_up(D, out_channels)   # single stream → D not 2D

    def forward(self, cam, lidar):
        n, _, h, w = cam.shape
        cam_b   = self.dropout_pre(self.proj_cam(cam))
        lidar_b = self.dropout_pre(self.proj_lidar(lidar))
        D = cam_b.shape[1]
        src_cam   = cam_b.view(n, D, -1).permute(2, 0, 1)
        src_lidar = lidar_b.view(n, D, -1).permute(2, 0, 1)
        attn1, _ = self.attn(self.norm1_cam(src_cam),
                             self.norm1_lidar(src_lidar),
                             self.norm1_lidar(src_lidar))
        cam_seq = src_cam + attn1
        cam_seq = cam_seq + self.ffn(self.norm2(cam_seq))
        return self.proj_up(cam_seq.permute(1, 2, 0).view(n, D, h, w))


# ─────────────────────────────────────────────────────────────────────────────
# A3: Unidirectional — LiDAR queries Camera only (Branch 2 only)
# Ablates: bidirectionality (removes Camera→LiDAR branch entirely)
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttnUniLidar(nn.Module):
    """A3: Q=lidar, K=V=cam — LiDAR attends to camera; no reverse branch."""
    def __init__(self, in_channels_cam, in_channels_lidar, out_channels,
                 num_heads=8, dropout=0.1, bottleneck_dim=256):
        super().__init__()
        D = bottleneck_dim
        self.proj_cam    = nn.Conv2d(in_channels_cam,   D, 1, bias=False)
        self.proj_lidar  = nn.Conv2d(in_channels_lidar, D, 1, bias=False)
        self.norm1_cam   = nn.LayerNorm(D)
        self.norm1_lidar = nn.LayerNorm(D)
        self.attn        = nn.MultiheadAttention(D, num_heads, dropout=dropout)
        self.norm2       = nn.LayerNorm(D)
        self.ffn         = _ffn(D, dropout)
        self.dropout_pre = nn.Dropout2d(p=min(dropout, 0.1))
        self.proj_up     = _proj_up(D, out_channels)   # single stream → D not 2D

    def forward(self, cam, lidar):
        n, _, h, w = cam.shape
        cam_b   = self.dropout_pre(self.proj_cam(cam))
        lidar_b = self.dropout_pre(self.proj_lidar(lidar))
        D = cam_b.shape[1]
        src_cam   = cam_b.view(n, D, -1).permute(2, 0, 1)
        src_lidar = lidar_b.view(n, D, -1).permute(2, 0, 1)
        attn2, _ = self.attn(self.norm1_lidar(src_lidar),
                             self.norm1_cam(src_cam),
                             self.norm1_cam(src_cam))
        lidar_seq = src_lidar + attn2
        lidar_seq = lidar_seq + self.ffn(self.norm2(lidar_seq))
        return self.proj_up(lidar_seq.permute(1, 2, 0).view(n, D, h, w))


# ─────────────────────────────────────────────────────────────────────────────
# A4: Camera passthrough — used for P3 and P4 in single-scale ablation
# Ablates: multi-scale fusion (P3/P4 receive camera only; only P5 fuses)
# ─────────────────────────────────────────────────────────────────────────────
class CameraPassthrough(nn.Module):
    """A4 helper: ignores LiDAR, returns camera features unchanged."""
    def __init__(self, *args, **kwargs):   # accept channel args so factory works
        super().__init__()

    def forward(self, cam, lidar):
        return cam   # LiDAR discarded — camera stream passes through unchanged


# ─────────────────────────────────────────────────────────────────────────────
# A5: No residual connections
# Ablates: both post-attention and post-FFN residual additions
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttnNoResidual(nn.Module):
    """A5: Full bidirectional attention + FFN but zero residual connections."""
    def __init__(self, in_channels_cam, in_channels_lidar, out_channels,
                 num_heads=8, dropout=0.1, bottleneck_dim=256):
        super().__init__()
        D = bottleneck_dim
        self.proj_cam    = nn.Conv2d(in_channels_cam,   D, 1, bias=False)
        self.proj_lidar  = nn.Conv2d(in_channels_lidar, D, 1, bias=False)
        self.norm1_cam   = nn.LayerNorm(D)
        self.norm1_lidar = nn.LayerNorm(D)
        self.attn_cam    = nn.MultiheadAttention(D, num_heads, dropout=dropout)
        self.attn_lidar  = nn.MultiheadAttention(D, num_heads, dropout=dropout)
        self.norm2_cam   = nn.LayerNorm(D)
        self.norm2_lidar = nn.LayerNorm(D)
        self.ffn_cam     = _ffn(D, dropout)
        self.ffn_lidar   = _ffn(D, dropout)
        self.dropout_pre = nn.Dropout2d(p=min(dropout, 0.1))
        self.proj_up     = _proj_up(D * 2, out_channels)

    def forward(self, cam, lidar):
        n, _, h, w = cam.shape
        cam_b   = self.dropout_pre(self.proj_cam(cam))
        lidar_b = self.dropout_pre(self.proj_lidar(lidar))
        D = cam_b.shape[1]
        src_cam   = cam_b.view(n, D, -1).permute(2, 0, 1)
        src_lidar = lidar_b.view(n, D, -1).permute(2, 0, 1)
        nc = self.norm1_cam(src_cam)
        nl = self.norm1_lidar(src_lidar)
        a1, _ = self.attn_cam(nc, nl, nl)
        cam_out   = self.ffn_cam(self.norm2_cam(a1))       # no src_cam + (no residual)
        a2, _ = self.attn_lidar(nl, nc, nc)
        lidar_out = self.ffn_lidar(self.norm2_lidar(a2))   # no src_lidar + (no residual)
        cam_sp   = cam_out.permute(1, 2, 0).view(n, D, h, w)
        lidar_sp = lidar_out.permute(1, 2, 0).view(n, D, h, w)
        return self.proj_up(torch.cat([cam_sp, lidar_sp], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# A6: No channel bottleneck — attention at full camera channel dimension
# Ablates: the 1x1 projection that reduces channels to bottleneck_dim=256
# NOTE: memory-intensive — P3 runs at 768-ch, P5 at 1536-ch attention
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttnNoBottleneck(nn.Module):
    """A6: Attention at full camera channel dim — no bottleneck projection."""
    def __init__(self, in_channels_cam, in_channels_lidar, out_channels,
                 num_heads=8, dropout=0.1, bottleneck_dim=None):   # bottleneck_dim ignored
        super().__init__()
        D = in_channels_cam   # 768 for P3/P4, 1536 for P5
        self.proj_lidar  = nn.Conv2d(in_channels_lidar, D, 1, bias=False) \
                           if in_channels_lidar != D else nn.Identity()
        # Adjust heads if D not divisible
        h = num_heads
        while D % h != 0 and h > 1:
            h -= 1
        self.norm1_cam   = nn.LayerNorm(D)
        self.norm1_lidar = nn.LayerNorm(D)
        self.attn_cam    = nn.MultiheadAttention(D, h, dropout=dropout)
        self.attn_lidar  = nn.MultiheadAttention(D, h, dropout=dropout)
        self.norm2_cam   = nn.LayerNorm(D)
        self.norm2_lidar = nn.LayerNorm(D)
        self.ffn_cam     = _ffn(D, dropout)
        self.ffn_lidar   = _ffn(D, dropout)
        self.proj_up     = _proj_up(D * 2, out_channels)
        self.D = D

    def forward(self, cam, lidar):
        n, _, h, w = cam.shape
        D = self.D
        lidar_p   = self.proj_lidar(lidar)
        src_cam   = cam.view(n, D, -1).permute(2, 0, 1)
        src_lidar = lidar_p.view(n, D, -1).permute(2, 0, 1)
        nc = self.norm1_cam(src_cam)
        nl = self.norm1_lidar(src_lidar)
        a1, _ = self.attn_cam(nc, nl, nl)
        cam_seq   = src_cam + a1
        cam_seq   = cam_seq + self.ffn_cam(self.norm2_cam(cam_seq))
        a2, _ = self.attn_lidar(nl, nc, nc)
        lidar_seq = src_lidar + a2
        lidar_seq = lidar_seq + self.ffn_lidar(self.norm2_lidar(lidar_seq))
        cam_sp   = cam_seq.permute(1, 2, 0).view(n, D, h, w)
        lidar_sp = lidar_seq.permute(1, 2, 0).view(n, D, h, w)
        return self.proj_up(torch.cat([cam_sp, lidar_sp], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# A7: No FFN block
# Ablates: the position-wise feed-forward network after each attention branch
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttnNoFFN(nn.Module):
    """A7: pre-LN → MHA → residual only — no FFN in either branch."""
    def __init__(self, in_channels_cam, in_channels_lidar, out_channels,
                 num_heads=8, dropout=0.1, bottleneck_dim=256):
        super().__init__()
        D = bottleneck_dim
        self.proj_cam    = nn.Conv2d(in_channels_cam,   D, 1, bias=False)
        self.proj_lidar  = nn.Conv2d(in_channels_lidar, D, 1, bias=False)
        self.norm1_cam   = nn.LayerNorm(D)
        self.norm1_lidar = nn.LayerNorm(D)
        self.attn_cam    = nn.MultiheadAttention(D, num_heads, dropout=dropout)
        self.attn_lidar  = nn.MultiheadAttention(D, num_heads, dropout=dropout)
        self.dropout_pre = nn.Dropout2d(p=min(dropout, 0.1))
        self.proj_up     = _proj_up(D * 2, out_channels)

    def forward(self, cam, lidar):
        n, _, h, w = cam.shape
        cam_b   = self.dropout_pre(self.proj_cam(cam))
        lidar_b = self.dropout_pre(self.proj_lidar(lidar))
        D = cam_b.shape[1]
        src_cam   = cam_b.view(n, D, -1).permute(2, 0, 1)
        src_lidar = lidar_b.view(n, D, -1).permute(2, 0, 1)
        nc = self.norm1_cam(src_cam)
        nl = self.norm1_lidar(src_lidar)
        a1, _ = self.attn_cam(nc, nl, nl)
        cam_seq   = src_cam + a1       # residual, no FFN
        a2, _ = self.attn_lidar(nl, nc, nc)
        lidar_seq = src_lidar + a2     # residual, no FFN
        cam_sp   = cam_seq.permute(1, 2, 0).view(n, D, h, w)
        lidar_sp = lidar_seq.permute(1, 2, 0).view(n, D, h, w)
        return self.proj_up(torch.cat([cam_sp, lidar_sp], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# A8: No spatial dropout
# Ablates: Dropout2D applied before the bottleneck projection
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttnNoDropout(nn.Module):
    """A8: Full MBiCMF minus the Dropout2D regularisation before projection."""
    def __init__(self, in_channels_cam, in_channels_lidar, out_channels,
                 num_heads=8, dropout=0.1, bottleneck_dim=256):
        super().__init__()
        D = bottleneck_dim
        self.proj_cam    = nn.Conv2d(in_channels_cam,   D, 1, bias=False)
        self.proj_lidar  = nn.Conv2d(in_channels_lidar, D, 1, bias=False)
        self.norm1_cam   = nn.LayerNorm(D)
        self.norm1_lidar = nn.LayerNorm(D)
        self.attn_cam    = nn.MultiheadAttention(D, num_heads, dropout=dropout)
        self.attn_lidar  = nn.MultiheadAttention(D, num_heads, dropout=dropout)
        self.norm2_cam   = nn.LayerNorm(D)
        self.norm2_lidar = nn.LayerNorm(D)
        self.ffn_cam     = _ffn(D, dropout)
        self.ffn_lidar   = _ffn(D, dropout)
        # NO Dropout2D
        self.proj_up     = _proj_up(D * 2, out_channels)

    def forward(self, cam, lidar):
        n, _, h, w = cam.shape
        # No Dropout2D before projection — all other ops identical to A1
        cam_b   = self.proj_cam(cam)
        lidar_b = self.proj_lidar(lidar)
        D = cam_b.shape[1]
        src_cam   = cam_b.view(n, D, -1).permute(2, 0, 1)
        src_lidar = lidar_b.view(n, D, -1).permute(2, 0, 1)
        nc = self.norm1_cam(src_cam)
        nl = self.norm1_lidar(src_lidar)
        a1, _ = self.attn_cam(nc, nl, nl)
        cam_seq   = src_cam + a1
        cam_seq   = cam_seq + self.ffn_cam(self.norm2_cam(cam_seq))
        a2, _ = self.attn_lidar(nl, nc, nc)
        lidar_seq = src_lidar + a2
        lidar_seq = lidar_seq + self.ffn_lidar(self.norm2_lidar(lidar_seq))
        cam_sp   = cam_seq.permute(1, 2, 0).view(n, D, h, w)
        lidar_sp = lidar_seq.permute(1, 2, 0).view(n, D, h, w)
        return self.proj_up(torch.cat([cam_sp, lidar_sp], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# A9: Simple concatenation fusion
# Ablates: the entire attention mechanism — shows baseline value of just merging
#          features spatially with no cross-modal reasoning.
# cat(cam, lidar) along channel dim → 1x1 Conv → GroupNorm → SiLU
# ─────────────────────────────────────────────────────────────────────────────
class ConcatFusion(nn.Module):
    """A9: Concatenate camera and LiDAR feature maps then project to output channels.
    No attention, no bottleneck, no FFN — the simplest possible fusion.
    Answers: does cross-attention add value beyond naive feature concatenation?"""
    def __init__(self, in_channels_cam, in_channels_lidar, out_channels,
                 num_heads=8, dropout=0.1, bottleneck_dim=256):
        # num_heads / dropout / bottleneck_dim accepted but unused (uniform interface)
        super().__init__()
        num_groups = 32 if out_channels % 32 == 0 else 1
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels_cam + in_channels_lidar, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, cam, lidar):
        return self.proj(torch.cat([cam, lidar], dim=1))
