"""
Stage 4 (Optional Refinement): CRF Depth Modulation + Hierarchical Loss
========================================================================
Two independent components that can be used together or separately:

A) CRFBoundaryRefiner
   - Takes a raw binary BEV occlusion mask M_occ^c  (nx, ny)
   - Uses depth-gradient image as the CRF pairwise term
   - Runs dense CRF mean-field iterations to sharpen / contract noisy boundaries
   - Returns refined soft mask  M_refined  ∈ [0, 1]

B) HierarchicalOSZLoss  (for training a learned boundary predictor)
   - Multi-scale boundary-aware loss combining:
       * Focal loss at full resolution  (handles extreme class imbalance)
       * Scale-invariant depth constraint (OBDCL-style, from MoDOT paper)
       * Laplacian pyramid boundary consistency  (coarse→fine)
   - Used to supervise a thin CNN that refines the geometric OSZ mask
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# A.  CRF Boundary Refiner (inference-time, no learnable params)
# ═══════════════════════════════════════════════════════════════════════════

class CRFBoundaryRefiner:
    """
    Lightweight dense CRF post-processor for OSZ masks.

    Energy function  E(x) = Σ_i θ_u(x_i)  +  Σ_{i≠j} θ_p(x_i, x_j)

    Unary term  θ_u :  from the raw geometric occlusion mask
    Pairwise   θ_p :  appearance kernel driven by BEV depth gradient
                       k(i,j) = w_α · exp(-|depth_i - depth_j|²/2σ_d²)
                                · exp(-|pos_i - pos_j|²/2σ_s²)

    Mean-field inference is run for `n_iters` steps.

    Args:
        sigma_d    : depth similarity bandwidth (metres)
        sigma_s    : spatial proximity bandwidth (BEV cells)
        w_alpha    : weight of the depth-appearance pairwise term
        n_iters    : number of mean-field iterations
        unary_conf : confidence weight of the unary (geometric) term
    """

    def __init__(
        self,
        sigma_d: float = 1.5,
        sigma_s: float = 3.0,
        w_alpha: float = 3.0,
        n_iters: int = 5,
        unary_conf: float = 2.0,
    ):
        self.sigma_d = sigma_d
        self.sigma_s = sigma_s
        self.w_alpha = w_alpha
        self.n_iters = n_iters
        self.unary_conf = unary_conf

    def _depth_gradient_bev(self, depth_bev: np.ndarray) -> np.ndarray:
        """
        Compute |∇depth| in BEV space.
        depth_bev: (nx, ny) float, average depth in each BEV cell (0 = unknown)
        """
        # Fill unknown cells with neighbour average before gradient
        mask = depth_bev > 0
        filled = depth_bev.copy()
        if mask.sum() > 0:
            # Simple nearest-neighbour fill via Gaussian smoothing
            num = gaussian_filter(filled * mask, sigma=2.0)
            den = gaussian_filter(mask.astype(float), sigma=2.0) + 1e-6
            filled = np.where(mask, filled, num / den)

        gx = np.gradient(filled, axis=0)
        gy = np.gradient(filled, axis=1)
        return np.sqrt(gx ** 2 + gy ** 2)

    def refine(
        self,
        mask_raw: np.ndarray,       # (nx, ny) bool or float in [0,1]
        depth_bev: np.ndarray,      # (nx, ny) float, depth per BEV cell
    ) -> np.ndarray:
        """
        Returns refined soft mask (nx, ny) float in [0, 1].
        """
        nx, ny = mask_raw.shape
        # Unary log-probabilities  (2-class: bg=0, shadow=1)
        p = mask_raw.astype(np.float32).clip(0.01, 0.99)
        # [2, nx, ny]: log P(x=0), log P(x=1)
        log_unary = np.stack([
            -self.unary_conf * p,
            -self.unary_conf * (1 - p)
        ], axis=0)

        depth_grad = self._depth_gradient_bev(depth_bev)  # (nx, ny)

        # Initialise Q = softmax of unary
        Q = np.exp(log_unary)
        Q /= Q.sum(axis=0, keepdims=True) + 1e-8

        for _ in range(self.n_iters):
            # ── Pairwise message passing via Gaussian blur approximation ─────
            # The bilateral filter is approximated as:
            #   message_k = Σ_j k(i,j) Q_k(j)
            # We use two separable Gaussian blurs:
            #   (1) spatial: blur Q with σ_s
            #   (2) depth-gated: weight by exp(-grad²/(2σ_d²)) then blur

            spatial_msg = np.stack([
                gaussian_filter(Q[k], sigma=self.sigma_s) for k in range(2)
            ], axis=0)

            depth_weight = np.exp(
                -depth_grad ** 2 / (2 * self.sigma_d ** 2)
            )
            depth_msg = np.stack([
                gaussian_filter(Q[k] * depth_weight, sigma=self.sigma_s / 2)
                for k in range(2)
            ], axis=0)

            # Pairwise compatibility (Potts model: penalise different labels)
            compat = np.array([[0, -1], [-1, 0]], dtype=np.float32)
            msg_combined = (
                spatial_msg +
                self.w_alpha * depth_msg
            )
            pairwise = np.einsum('kl,l...->k...', compat, msg_combined)

            # Update Q
            log_Q = log_unary + pairwise
            Q = np.exp(log_Q - log_Q.max(axis=0, keepdims=True))
            Q /= Q.sum(axis=0, keepdims=True) + 1e-8

        return Q[1]   # probability of shadow class


# ═══════════════════════════════════════════════════════════════════════════
# B.  Hierarchical OSZ Loss  (for training a refinement CNN)
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalOSZLoss(nn.Module):
    """
    Three-component hierarchical loss for supervised OSZ boundary refinement.

    L_total = λ_focal · L_focal
            + λ_depth · L_depth_constraint
            + λ_lap   · L_laplacian

    Components:
    -----------
    L_focal  : Focal loss on boundary pixels (α-balanced, γ=2)
               Handles extreme foreground (shadow) / background imbalance.
               Operates at full resolution.

    L_depth  : OB-Depth Constraint Loss (OBDCL, inspired by MoDOT).
               Enforces that predicted depth MUST be discontinuous across
               predicted boundary pixels:
                   L_depth = mean( max(0, margin - |d_i - d_j|) )
               where (i,j) are predicted boundary pixel pairs.

    L_lap    : Laplacian pyramid boundary consistency.
               Computes boundary maps at 3 scales and averages losses,
               so coarse structure is learnt first (curriculum effect).

    Args:
        lambda_focal : weight for focal loss
        lambda_depth : weight for depth constraint loss
        lambda_lap   : weight for laplacian pyramid loss
        focal_alpha  : focal loss α (shadow class weight)
        focal_gamma  : focal loss γ
        depth_margin : minimum depth discontinuity required at boundaries (m)
        n_scales     : number of pyramid levels for laplacian loss
    """

    def __init__(
        self,
        lambda_focal: float = 1.0,
        lambda_depth: float = 0.5,
        lambda_lap:   float = 0.3,
        focal_alpha:  float = 0.75,
        focal_gamma:  float = 2.0,
        depth_margin: float = 0.5,
        n_scales:     int   = 3,
    ):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.lambda_depth = lambda_depth
        self.lambda_lap   = lambda_lap
        self.focal_alpha  = focal_alpha
        self.focal_gamma  = focal_gamma
        self.depth_margin = depth_margin
        self.n_scales     = n_scales

        # Laplacian kernel (fixed, no grad)
        lap_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

    # ── Focal Loss ──────────────────────────────────────────────────────────
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, 1, H, W), pred is logit (before sigmoid).
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-bce)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal = alpha_t * (1 - p_t) ** self.focal_gamma * bce
        return focal.mean()

    # ── OB-Depth Constraint Loss ─────────────────────────────────────────────
    def _depth_constraint_loss(
        self,
        pred_boundary: torch.Tensor,   # (B, 1, H, W) soft boundary [0,1]
        depth_bev: torch.Tensor,        # (B, 1, H, W) BEV depth map
    ) -> torch.Tensor:
        """
        At predicted boundary pixels, enforce depth discontinuity
        between adjacent cells (horizontal + vertical neighbours).
        """
        # Absolute depth difference between adjacent pixels
        diff_x = (depth_bev[:, :, :, 1:] - depth_bev[:, :, :, :-1]).abs()
        diff_y = (depth_bev[:, :, 1:, :] - depth_bev[:, :, :-1, :]).abs()

        # Boundary confidence at those positions (use minimum of pair)
        b_x = pred_boundary[:, :, :, 1:].min(pred_boundary[:, :, :, :-1])
        b_y = pred_boundary[:, :, 1:, :].min(pred_boundary[:, :, :-1, :])

        # Hinge: penalise if depth diff < margin at boundary pixels
        loss_x = b_x * F.relu(self.depth_margin - diff_x)
        loss_y = b_y * F.relu(self.depth_margin - diff_y)

        return (loss_x.mean() + loss_y.mean()) / 2

    # ── Laplacian Pyramid Loss ───────────────────────────────────────────────
    def _laplacian_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge map via Laplacian. x: (B, 1, H, W)"""
        return F.conv2d(x, self.lap_kernel, padding=1).abs()

    def _laplacian_pyramid_loss(
        self,
        pred: torch.Tensor,    # (B, 1, H, W) logit
        target: torch.Tensor,  # (B, 1, H, W) binary GT
    ) -> torch.Tensor:
        pred_sig = torch.sigmoid(pred)
        total = torch.zeros(1, device=pred.device)
        p = pred_sig
        t = target
        for s in range(self.n_scales):
            # Boundary edge maps at this scale
            p_edge = self._laplacian_boundary(p)
            t_edge = self._laplacian_boundary(t)
            # L1 between edge strengths — encourages sharp, coincident boundaries
            scale_loss = F.l1_loss(p_edge, t_edge)
            total = total + scale_loss / (2 ** s)   # coarser scale = less weight
            # Downsample for next level
            if s < self.n_scales - 1:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
        return total / self.n_scales

    # ── Combined Forward ────────────────────────────────────────────────────
    def forward(
        self,
        pred_logit: torch.Tensor,       # (B, 1, H, W) raw network output
        target_mask: torch.Tensor,      # (B, 1, H, W) GT OSZ binary mask
        depth_bev: Optional[torch.Tensor] = None,   # (B, 1, H, W)
    ) -> dict:
        import collections
        losses = collections.OrderedDict()

        losses['focal'] = self._focal_loss(pred_logit, target_mask)

        losses['laplacian'] = self._laplacian_pyramid_loss(pred_logit, target_mask)

        if depth_bev is not None:
            pred_boundary = torch.sigmoid(pred_logit).detach()  # don't backprop through boundary
            losses['depth_constraint'] = self._depth_constraint_loss(
                pred_boundary, depth_bev
            )
        else:
            losses['depth_constraint'] = torch.zeros(1, device=pred_logit.device)[0]

        total = (
            self.lambda_focal * losses['focal'] +
            self.lambda_lap   * losses['laplacian'] +
            self.lambda_depth * losses['depth_constraint']
        )
        losses['total'] = total
        return losses


# ═══════════════════════════════════════════════════════════════════════════
# C.  Lightweight Refinement CNN  (thin U-Net style, 3 encoder levels)
# ═══════════════════════════════════════════════════════════════════════════

class OSZRefineCNN(nn.Module):
    """
    Thin encoder-decoder that takes:
        - geometric OSZ mask (1 channel)
        - BEV depth map       (1 channel)
        - BEV occupancy       (1 channel, optional)
    and outputs a refined boundary logit map (1 channel).

    Total params: ~150K — fast to train, easy to overfit-check.
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()

        def conv_bn_relu(cin, cout, k=3, p=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, padding=p, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = nn.Sequential(conv_bn_relu(in_channels, 32), conv_bn_relu(32, 32))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), conv_bn_relu(32, 64), conv_bn_relu(64, 64))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), conv_bn_relu(64, 128), conv_bn_relu(128, 128))

        # Bottleneck (strip convolution to capture elongated OB geometry)
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            conv_bn_relu(128, 256),
            nn.Conv2d(256, 256, (1, 7), padding=(0, 3), bias=False),  # horizontal strip
            nn.Conv2d(256, 256, (7, 1), padding=(3, 0), bias=False),  # vertical strip
            nn.ReLU(inplace=True),
        )

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_bn_relu(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_bn_relu(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_bn_relu(64, 32)

        # Output: boundary logit
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) → logit: (B, 1, H, W)"""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
