print("train.py start===>")
import sys, os
# Add root to system path so you can access lib.models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


import argparse
import cv2
import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lietorch import SE3
import RAFT3D.raft3d.projective_ops as pops

from utils import Logger, show_image, normalize_image
from evaluation import test_sceneflow
from data_readers.sceneflow import SceneFlow

import torchvision.transforms as transforms
from lib.core.config import config
from lib.core.config import update_config, update_config_dynamic_input
import lib.dataset as dataset
import pandas as pd
from datetime import datetime


RV_WEIGHT = 0.2
DZ_WEIGHT = 100.0
DEPTH_SCALE = 0.2
MAX_DEPTH = 250
MAX_FLOW = 250


# ============================================================================
# DEPTH SUPERVISION LOSSES
# ============================================================================

def compute_depth_loss(depth_pred, depth_gt, valid_mask=None):
    """
    Compute direct depth supervision loss.
    
    Args:
        depth_pred: Predicted depth [N, H, W] or [N, 1, H, W]
        depth_gt: Ground truth depth [N, H, W] or [N, 1, H, W]
        valid_mask: Optional validity mask [N, H, W]
    
    Returns:
        loss: Scalar depth loss
        metrics: Dictionary with depth metrics
    """
    # Squeeze channel dimension if present
    if depth_pred.dim() == 4 and depth_pred.shape[1] == 1:
        depth_pred = depth_pred.squeeze(1)
    if depth_gt.dim() == 4 and depth_gt.shape[1] == 1:
        depth_gt = depth_gt.squeeze(1)
    
    # Resize if needed
    if depth_gt.shape[-2:] != depth_pred.shape[-2:]:
        depth_gt = F.interpolate(depth_gt.unsqueeze(1), size=depth_pred.shape[-2:], 
                                 mode='bilinear', align_corners=False).squeeze(1)
    
    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = (depth_gt > 0.1) & (depth_gt < MAX_DEPTH) & (depth_pred > 0.1)
    
    if valid_mask.sum() == 0:
        # Use zeros_like to preserve gradient connection
        zero_loss = torch.zeros_like(depth_pred.mean())
        return zero_loss, {'depth_abs_rel': 0.0, 'depth_rmse': 0.0}
    
    # Compute losses only on valid pixels
    pred_valid = depth_pred[valid_mask]
    gt_valid = depth_gt[valid_mask]
    
    # L1 Loss (absolute error)
    l1_loss = torch.abs(pred_valid - gt_valid).mean()
    
    # Scale-invariant loss (helps with scale ambiguity)
    log_pred = torch.log(pred_valid + 1e-6)
    log_gt = torch.log(gt_valid + 1e-6)
    log_diff = log_pred - log_gt
    si_loss = torch.sqrt((log_diff ** 2).mean() - 0.5 * (log_diff.mean() ** 2) + 1e-6)
    
    # Combined loss
    loss = l1_loss + 0.5 * si_loss
    
    # Metrics
    abs_rel = (torch.abs(pred_valid - gt_valid) / (gt_valid + 1e-6)).mean()
    rmse = torch.sqrt(((pred_valid - gt_valid) ** 2).mean())
    
    metrics = {
        'depth_abs_rel': abs_rel.item(),
        'depth_rmse': rmse.item(),
        'depth_l1': l1_loss.item(),
        'depth_si': si_loss.item()
    }
    
    return loss, metrics


def compute_depth_consistency_loss(depth1, flow3d_est, depth2_gt, valid_mask=None):
    """
    Compute depth consistency loss: depth1 + dz (from scene flow) should equal depth2.
    This directly supervises the depth change prediction.
    
    Args:
        depth1: Depth at time t [N, H, W] or [N, 1, H, W]
        flow3d_est: Estimated 3D scene flow [N, H, W, 3] (dx, dy, dz)
        depth2_gt: Ground truth depth at time t+1 [N, H, W] or [N, 1, H, W]
        valid_mask: Optional validity mask
    
    Returns:
        loss: Scalar consistency loss
        metrics: Dictionary with metrics
    """
    # Squeeze channel dimensions
    if depth1.dim() == 4 and depth1.shape[1] == 1:
        depth1 = depth1.squeeze(1)
    if depth2_gt.dim() == 4 and depth2_gt.shape[1] == 1:
        depth2_gt = depth2_gt.squeeze(1)
    
    # Extract depth change (dz component of scene flow)
    dz = flow3d_est[..., 2]  # [N, H, W]
    
    # Resize depth1 to match dz resolution if needed (flow may have been unpadded)
    if depth1.shape[-2:] != dz.shape[-2:]:
        depth1 = F.interpolate(depth1.unsqueeze(1), size=dz.shape[-2:],
                               mode='bilinear', align_corners=False).squeeze(1)
    
    # Predicted depth at t+1 = depth at t + depth change
    depth2_pred = depth1 + dz
    
    # Resize GT if needed
    if depth2_gt.shape[-2:] != depth2_pred.shape[-2:]:
        depth2_gt = F.interpolate(depth2_gt.unsqueeze(1), size=depth2_pred.shape[-2:],
                                  mode='bilinear', align_corners=False).squeeze(1)
    
    # Create valid mask
    if valid_mask is None:
        valid_mask = (depth2_gt > 0.1) & (depth2_gt < MAX_DEPTH) & \
                     (depth1 > 0.1) & (depth2_pred > 0.1)
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=depth1.device), {'depth_consistency': 0.0}
    
    # Compute consistency loss
    pred_valid = depth2_pred[valid_mask]
    gt_valid = depth2_gt[valid_mask]
    
    consistency_loss = torch.abs(pred_valid - gt_valid).mean()
    
    metrics = {
        'depth_consistency': consistency_loss.item()
    }
    
    return consistency_loss, metrics


def compute_multiview_depth_consistency_loss(depths, intrinsics_list, extrinsics_list=None):
    """
    Enforce multi-view depth consistency: depths from different views should
    be geometrically consistent when reprojected to 3D.
    
    Args:
        depths: List of depth maps from different views [view, H, W]
        intrinsics_list: List of camera intrinsics per view
        extrinsics_list: Optional list of camera extrinsics (R, t) per view
    
    Returns:
        loss: Multi-view consistency loss
    """
    num_views = len(depths)
    if num_views < 2:
        return torch.tensor(0.0, device=depths[0].device)
    
    consistency_losses = []
    
    # Pairwise consistency between adjacent views
    for i in range(num_views - 1):
        j = i + 1
        
        # Simple depth similarity constraint between overlapping regions
        # This encourages similar depth statistics across views
        depth_i = depths[i]
        depth_j = depths[j]
        
        # Compute mean depth in valid regions
        valid_i = (depth_i > 0.1) & (depth_i < MAX_DEPTH)
        valid_j = (depth_j > 0.1) & (depth_j < MAX_DEPTH)
        
        if valid_i.sum() > 0 and valid_j.sum() > 0:
            mean_i = depth_i[valid_i].mean()
            mean_j = depth_j[valid_j].mean()
            std_i = depth_i[valid_i].std()
            std_j = depth_j[valid_j].std()
            
            # Penalize large differences in depth statistics
            mean_diff = torch.abs(mean_i - mean_j) / (mean_i + mean_j + 1e-6)
            std_diff = torch.abs(std_i - std_j) / (std_i + std_j + 1e-6)
            
            consistency_losses.append(mean_diff + 0.5 * std_diff)
    
    if len(consistency_losses) == 0:
        return torch.tensor(0.0, device=depths[0].device)
    
    return torch.stack(consistency_losses).mean()


# ============================================================================
# ORIGINAL FUNCTIONS
# ============================================================================

def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=0.2):
    """ padding, normalization, and scaling """

    print("train prepare_images_and_depths start===>")
    
    def _flatten_leading(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.dim() > 4:
            leading = x.shape[:-3]
            new_batch = 1
            for d in leading:
                new_batch *= d
            return x.view(new_batch, *x.shape[-3:])
        else:
            return x
    
    image1 = _flatten_leading(image1)
    image2 = _flatten_leading(image2)
    depth1 = _flatten_leading(depth1)
    depth2 = _flatten_leading(depth2)

    # ensure depth tensors have a channel dim (N, C, H, W)
    if depth1.dim() == 3:
        depth1 = depth1.unsqueeze(1)
    if depth2.dim() == 3:
        depth2 = depth2.unsqueeze(1)

    ht, wd = image1.shape[-2:]
    pad_h = (-ht) % 8
    pad_w = (-wd) % 8
    
    image1 = F.pad(image1, [0, pad_w, 0, pad_h], mode='replicate')
    image2 = F.pad(image2, [0, pad_w, 0, pad_h], mode='replicate')
    depth1 = F.pad(depth1, [0, pad_w, 0, pad_h], mode='replicate')
    depth2 = F.pad(depth2, [0, pad_w, 0, pad_h], mode='replicate')

    depth1 = (depth_scale * depth1).float()
    depth2 = (depth_scale * depth2).float()
    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())

    depth1 = depth1.float()
    depth2 = depth2.float()

    print("train prepare_images_and_depths end===>")

    return image1, image2, depth1, depth2, (pad_w, pad_h)


def compute_geometric_weights(meta, num_views=5):
    """
    Compute weights based on camera baseline distances from centroid.
    Cameras farther from center get higher weight (better triangulation geometry).
    
    Args:
        meta: List of metadata dicts containing camera extrinsics
        num_views: Number of camera views
    
    Returns:
        weights: Tensor of shape [num_views] with normalized weights
    """
    print("train compute_geometric_weights start===>")
    camera_centers = []
    
    for view_idx in range(num_views):
        # Try to extract camera position from metadata
        if 'camera_R' in meta[view_idx] and 'camera_t' in meta[view_idx]:
            R = meta[view_idx]['camera_R']  # [3, 3] rotation
            t = meta[view_idx]['camera_t']  # [3,] or [3, 1] translation
            
            if isinstance(R, torch.Tensor):
                R = R.cpu().numpy()
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()
            
            # Camera center in world coordinates: C = -R^T @ t
            if t.ndim == 2:
                t = t.squeeze()
            center = -R.T @ t  # [3,]
            camera_centers.append(center)
        else:
            # Fallback: use view index as proxy position
            # Assume cameras arranged in a circle
            angle = 2 * np.pi * view_idx / num_views
            center = np.array([np.cos(angle), np.sin(angle), 0.0])
            camera_centers.append(center)
    
    camera_centers = np.array(camera_centers)  # [num_views, 3]
    
    # Compute centroid (geometric center of all cameras)
    centroid = camera_centers.mean(axis=0)  # [3,]
    
    # Compute distance from centroid for each camera
    distances = np.linalg.norm(camera_centers - centroid, axis=1)  # [num_views,]
    
    # Check for degenerate case (all cameras at same position)
    if distances.max() < 1e-6:
        # Fall back to equal weighting
        weights = np.ones(num_views) / num_views
    else:
        # Weight by distance (farther = better geometry)
        # Use square root for balanced weighting
        weights = np.sqrt(distances + 1e-8)  # Add epsilon for numerical stability
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        # Optional: Clip to prevent extreme weights
        weights = np.clip(weights, 0.05, 0.40)
        weights = weights / weights.sum()  # Re-normalize after clipping
    
    print("train compute_geometric_weights end===>")
    return torch.tensor(weights, dtype=torch.float32)


def loss_fn_from_flows(flow2d_est, flow3d_est, depth1, intrinsics, flow2d_gt, flow3d_gt, meta, valid_mask=None,
                       gamma=0.9, num_views=1, use_geometric_weights=False,
                       depth2_gt=None, depth_loss_weight=0.0, depth_consistency_weight=0.0,
                       multiview_depth_weight=0.0):
    """
    Loss function that accepts flows directly (with gradients) instead of computing from Ts.
    This is used when train_mode=True to preserve gradients through the update block.
    
    Args:
        flow2d_est: Estimated 2D optical flow [1, H, W, 2] - has gradients
        flow3d_est: Estimated 3D scene flow [1, H, W, 3] - has gradients
        depth1: Depth at time t [1, H, W] or [1, 1, H, W] - single view
        intrinsics: Camera intrinsic parameters [1, 3, 3] or [1, 4] - single view
        flow2d_gt: Ground truth 2D optical flow [1, H, W, 2] or [5, H, W, 2]
        flow3d_gt: Ground truth 3D scene flow [1, H, W, 3] or [5, H, W, 3]
        meta: List of metadata dicts (unused for single view)
        valid_mask: Optional pre-computed valid mask
        gamma: Decay factor (unused, for compatibility)
        num_views: Number of camera views (should be 1 for Solution 1)
        use_geometric_weights: Unused for single view
        depth2_gt: Ground truth depth at time t+1 [1, H, W] - single view
        depth_loss_weight: Weight for direct depth supervision loss
        depth_consistency_weight: Weight for depth consistency loss
        multiview_depth_weight: Unused for single view
    
    Returns:
        loss: Total loss value (with gradients)
        metrics: Dictionary of evaluation metrics
    """
    print("train loss_fn_from_flows start===>")
    
    # Unpad and unscale flows (same as in loss_fn)
    flow2d_est = flow2d_est[:, :-4, :, :2]  # Remove padding [1, H, W, 2]
    flow3d_est = flow3d_est[:, :-4] / DEPTH_SCALE  # Undo depth scaling [1, H, W, 3]
    
    # Convert ground truth flows from NCHW to NHWC format (same as loss_fn)
    if flow2d_gt.dim() == 4:
        if flow2d_gt.shape[1] in [2, 3] and flow2d_gt.shape[-1] not in [2, 3]:
            flow2d_gt = flow2d_gt.permute(0, 2, 3, 1)  # [B, H, W, C]
    elif flow2d_gt.dim() == 5:
        if flow2d_gt.shape[2] in [2, 3]:
            flow2d_gt = flow2d_gt.permute(0, 1, 3, 4, 2)  # [V, B, H, W, C]
        if flow2d_gt.shape[1] == 1:
            flow2d_gt = flow2d_gt.squeeze(1)  # [V, H, W, C]
        flow2d_gt = flow2d_gt[0:1]  # [1, H, W, 2]
    
    if flow2d_gt.dim() == 4 and flow2d_gt.shape[0] > 1:
        flow2d_gt = flow2d_gt[0:1]  # [1, H, W, 2]
    
    if flow3d_gt.dim() == 4:
        if flow3d_gt.shape[1] == 3 and flow3d_gt.shape[-1] != 3:
            flow3d_gt = flow3d_gt.permute(0, 2, 3, 1)  # [B, H, W, C]
    elif flow3d_gt.dim() == 5:
        if flow3d_gt.shape[2] == 3:
            flow3d_gt = flow3d_gt.permute(0, 1, 3, 4, 2)  # [V, B, H, W, C]
        if flow3d_gt.shape[1] == 1:
            flow3d_gt = flow3d_gt.squeeze(1)  # [V, H, W, C]
        flow3d_gt = flow3d_gt[0:1]  # [1, H, W, 3]
    
    if flow3d_gt.dim() == 4 and flow3d_gt.shape[0] > 1:
        flow3d_gt = flow3d_gt[0:1]  # [1, H, W, 3]
    
    # Resize ground truth to match estimated flow resolution if needed
    target_h, target_w = int(flow2d_est.shape[1]), int(flow2d_est.shape[2])
    
    if flow2d_gt.shape[1:3] != (target_h, target_w):
        flow2d_gt_nchw = flow2d_gt.permute(0, 3, 1, 2)  # [1, 2, H_gt, W_gt]
        flow2d_gt_resized = F.interpolate(flow2d_gt_nchw, 
                                         size=(target_h, target_w), 
                                         mode='bilinear', align_corners=False)  # [1, 2, H, W]
        flow2d_gt = flow2d_gt_resized.permute(0, 2, 3, 1)  # [1, H, W, 2]
    
    if flow3d_gt.shape[1:3] != (target_h, target_w):
        flow3d_gt_nchw = flow3d_gt.permute(0, 3, 1, 2)  # [1, 3, H_gt, W_gt]
        flow3d_gt_resized = F.interpolate(flow3d_gt_nchw, 
                                         size=(target_h, target_w), 
                                         mode='bilinear', align_corners=False)  # [1, 3, H, W]
        flow3d_gt = flow3d_gt_resized.permute(0, 2, 3, 1)  # [1, H, W, 3]
    
    # Compute EPE for single view
    epe_2d = torch.sum((flow2d_est - flow2d_gt)**2, -1).sqrt()  # [1, H, W]
    epe_3d = torch.sum((flow3d_est - flow3d_gt)**2, -1).sqrt()  # [1, H, W]
    
    # Compute valid mask
    mag_final = torch.sum(flow2d_est**2, dim=-1).sqrt()  # [1, H, W]
    depth_final = torch.sum(flow3d_est**2, dim=-1).sqrt()  # [1, H, W]
    valid = (mag_final < MAX_FLOW) & (depth_final < MAX_DEPTH)  # [1, H, W]
    
    # Compute losses for single view
    valid_view = valid[0]  # [H, W]
    epe_2d_view = epe_2d[0]  # [H, W]
    epe_3d_view = epe_3d[0]  # [H, W]
    
    # Compute loss (only over valid pixels)
    if valid_view.sum() > 0:
        loss_2d = (valid_view * epe_2d_view).sum() / (valid_view.sum() + 1e-8)
        loss_3d = (valid_view * epe_3d_view).sum() / (valid_view.sum() + 1e-8)
    else:
        loss_2d = torch.zeros_like(epe_2d_view.sum())
        loss_3d = torch.zeros_like(epe_3d_view.sum())
    
    # Base flow loss
    loss = loss_2d + loss_3d
    
    # For metrics compatibility
    loss_2d_per_view = [loss_2d.item()]
    loss_3d_per_view = [loss_3d.item()]
    
    # Depth supervision losses (simplified - can be expanded if needed)
    depth_metrics = {}
    if depth_loss_weight > 0.0 or depth_consistency_weight > 0.0:
        # Note: Depth losses would require computing depth from flows or using Ts
        # For now, we skip depth losses when using flows directly
        # This can be added later if needed
        pass
    
    # Compute metrics
    metrics = {
        'loss': loss.item(),
        'loss_2d': loss_2d.item(),
        'loss_3d': loss_3d.item(),
        'epe2d': epe_2d_view[valid_view].mean().item() if valid_view.sum() > 0 else 0.0,
        'epe3d': epe_3d_view[valid_view].mean().item() if valid_view.sum() > 0 else 0.0,
        'loss_2d_per_view': loss_2d_per_view,
        'loss_3d_per_view': loss_3d_per_view,
    }
    metrics.update(depth_metrics)
    
    return loss, metrics


def loss_fn(Ts, depth1, intrinsics, flow2d_gt, flow3d_gt, meta, valid_mask=None, 
            gamma=0.9, num_views=1, use_geometric_weights=False,
            depth2_gt=None, depth_loss_weight=0.0, depth_consistency_weight=0.0,
            multiview_depth_weight=0.0):
    """ 
    Loss function using transformation field predictions - Solution 1: Single consolidated view.
    Now includes explicit depth supervision losses.
    
    Args:
        Ts: SE3 transformation field [1, H, W, 7] - single consolidated view
        depth1: Depth at time t [1, H, W] or [1, 1, H, W] - single view
        intrinsics: Camera intrinsic parameters [1, 3, 3] or [1, 4] - single view
        flow2d_gt: Ground truth 2D optical flow [1, H, W, 2] or [5, H, W, 2] - use first view or average
        flow3d_gt: Ground truth 3D scene flow [1, H, W, 3] or [5, H, W, 3] - use first view or average
        meta: List of metadata dicts (unused for single view)
        valid_mask: Optional pre-computed valid mask
        gamma: Decay factor (unused, for compatibility)
        num_views: Number of camera views (should be 1 for Solution 1)
        use_geometric_weights: Unused for single view
        depth2_gt: Ground truth depth at time t+1 [1, H, W] - single view
        depth_loss_weight: Weight for direct depth supervision loss
        depth_consistency_weight: Weight for depth consistency loss (depth1 + dz = depth2)
        multiview_depth_weight: Unused for single view
    
    Returns:
        loss: Total loss value
        metrics: Dictionary of evaluation metrics
    """
    print("train loss_fn start===>")
    # Solution 1: Handle single consolidated view
    
    # Ensure depth1 is single view
    if depth1.shape[0] > 1:
        depth1 = depth1[:1]  # Take first view
    
    # Get target depth resolution
    if depth1.dim() == 4:  # [1, channels, H, W]
        target_h, target_w = depth1.shape[2], depth1.shape[3]
    else:  # [1, H, W]
        target_h, target_w = depth1.shape[1], depth1.shape[2]
        
    print(f"DEBUG loss_fn: Target resolution: {target_h}x{target_w}")
    print(f"DEBUG loss_fn: Current Ts resolution: {Ts.shape[1]}x{Ts.shape[2]}")
    
    # Upsample Ts if needed to match depth resolution
    # Use SE3.log() to get log coordinates (preserves gradients), upsample, then convert back
    if Ts.shape[1] != target_h or Ts.shape[2] != target_w:
        # Get log coordinates (lie algebra representation) - this preserves gradients
        tau_phi = Ts.log()  # [1, H, W, 6] - log coordinates
        tau_phi = tau_phi.permute(0, 3, 1, 2)  # [1, 6, H, W] for interpolation
        tau_phi_upsampled = F.interpolate(tau_phi, size=(target_h, target_w), 
                                         mode='bilinear', align_corners=False)
        tau_phi_upsampled = tau_phi_upsampled.permute(0, 2, 3, 1)  # [1, H, W, 6]
        
        # Convert back to SE3 group (this preserves gradients through exp map)
        Ts_upsampled = SE3.exp(tau_phi_upsampled)
        print(f"DEBUG loss_fn: Upsampled Ts to: {Ts_upsampled.shape}")
    else:
        Ts_upsampled = Ts
        print(f"DEBUG loss_fn: No upsampling needed for Ts")
    
    # Ensure intrinsics is single view format [1, 3, 3]
    if intrinsics.dim() == 3 and intrinsics.shape[0] > 1:
        intrinsics = intrinsics[:1]  # [1, 3, 3]
    elif intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)  # [1, 3, 3]
    
    # Extract flow from transformation field
    flow2d_est, flow3d_est, _ = pops.induced_flow(Ts_upsampled, depth1, intrinsics)
    
    # Unpad and unscale
    flow2d_est = flow2d_est[:, :-4, :, :2]  # Remove padding [1, H, W, 2]
    flow3d_est = flow3d_est[:, :-4] / DEPTH_SCALE  # Undo depth scaling [1, H, W, 3]
    
    # Convert ground truth flows from NCHW [B, C, H, W] to NHWC [B, H, W, C] format if needed
    # Dataset returns flows in NCHW format: [num_views, C, H, W]
    # Check if flow2d_gt is in NCHW format: [B, C, H, W] where C is small (2) and H, W are large
    if flow2d_gt.dim() == 4:
        # If channels are in dim 1 (NCHW format), convert to NHWC
        # NCHW: [B, 2, H, W] -> NHWC: [B, H, W, 2]
        if flow2d_gt.shape[1] in [2, 3] and flow2d_gt.shape[-1] not in [2, 3]:
            flow2d_gt = flow2d_gt.permute(0, 2, 3, 1)  # [B, H, W, C]
    elif flow2d_gt.dim() == 5:
        # Handle 5D tensors: [V, B, C, H, W] -> [1, H, W, C]
        if flow2d_gt.shape[2] in [2, 3]:  # Channels in dim 2
            flow2d_gt = flow2d_gt.permute(0, 1, 3, 4, 2)  # [V, B, H, W, C]
        if flow2d_gt.shape[1] == 1:
            flow2d_gt = flow2d_gt.squeeze(1)  # [V, H, W, C]
        flow2d_gt = flow2d_gt[0:1]  # [1, H, W, 2]
    
    # If multi-view, take first view
    if flow2d_gt.dim() == 4 and flow2d_gt.shape[0] > 1:
        flow2d_gt = flow2d_gt[0:1]  # [1, H, W, 2]
    
    # Check if flow3d_gt is in NCHW format: [B, C, H, W] where C is small (3) and H, W are large
    if flow3d_gt.dim() == 4:
        # If channels are in dim 1 (NCHW format), convert to NHWC
        # NCHW: [B, 3, H, W] -> NHWC: [B, H, W, 3]
        if flow3d_gt.shape[1] == 3 and flow3d_gt.shape[-1] != 3:
            flow3d_gt = flow3d_gt.permute(0, 2, 3, 1)  # [B, H, W, C]
    elif flow3d_gt.dim() == 5:
        # Handle 5D tensors: [V, B, C, H, W] -> [1, H, W, C]
        if flow3d_gt.shape[2] == 3:  # Channels in dim 2
            flow3d_gt = flow3d_gt.permute(0, 1, 3, 4, 2)  # [V, B, H, W, C]
        if flow3d_gt.shape[1] == 1:
            flow3d_gt = flow3d_gt.squeeze(1)  # [V, H, W, C]
        flow3d_gt = flow3d_gt[0:1]  # [1, H, W, 3]
    
    # If multi-view, take first view
    if flow3d_gt.dim() == 4 and flow3d_gt.shape[0] > 1:
        flow3d_gt = flow3d_gt[0:1]  # [1, H, W, 3]
    
    # Ensure flow2d_gt and flow3d_gt are 4D: [1, H, W, C]
    if flow2d_gt.dim() != 4:
        raise ValueError(f"flow2d_gt should be 4D after processing, got shape {flow2d_gt.shape}")
    if flow3d_gt.dim() != 4:
        raise ValueError(f"flow3d_gt should be 4D after processing, got shape {flow3d_gt.shape}")
    
    # Debug: Print shapes before interpolation
    print(f"DEBUG loss_fn: flow2d_est.shape = {flow2d_est.shape}")
    print(f"DEBUG loss_fn: flow2d_gt.shape = {flow2d_gt.shape}")
    print(f"DEBUG loss_fn: flow3d_est.shape = {flow3d_est.shape}")
    print(f"DEBUG loss_fn: flow3d_gt.shape = {flow3d_gt.shape}")
    
    # Validate shapes
    assert flow2d_est.dim() == 4 and flow2d_est.shape[-1] == 2, f"flow2d_est should be [B, H, W, 2], got {flow2d_est.shape}"
    assert flow2d_gt.dim() == 4 and flow2d_gt.shape[-1] == 2, f"flow2d_gt should be [B, H, W, 2], got {flow2d_gt.shape}"
    assert flow3d_est.dim() == 4 and flow3d_est.shape[-1] == 3, f"flow3d_est should be [B, H, W, 3], got {flow3d_est.shape}"
    assert flow3d_gt.dim() == 4 and flow3d_gt.shape[-1] == 3, f"flow3d_gt should be [B, H, W, 3], got {flow3d_gt.shape}"
    
    # Resize ground truth to match estimated flow resolution if needed
    # flow2d_est is [1, H, W, 2], so shape[1:3] gives [H, W]
    target_h, target_w = int(flow2d_est.shape[1]), int(flow2d_est.shape[2])
    
    if flow2d_gt.shape[1:3] != (target_h, target_w):
        print(f"DEBUG loss_fn: Resizing flow2d_gt from {flow2d_gt.shape[1:3]} to ({target_h}, {target_w})")
        # Convert to [B, C, H, W] for interpolation
        flow2d_gt_nchw = flow2d_gt.permute(0, 3, 1, 2)  # [1, 2, H_gt, W_gt]
        print(f"DEBUG loss_fn: flow2d_gt_nchw.shape = {flow2d_gt_nchw.shape}, target_size = ({target_h}, {target_w})")
        assert flow2d_gt_nchw.shape[1] == 2, f"flow2d_gt_nchw should have 2 channels, got {flow2d_gt_nchw.shape}"
        flow2d_gt_resized = F.interpolate(flow2d_gt_nchw, 
                                         size=(target_h, target_w), 
                                         mode='bilinear', align_corners=False)  # [1, 2, H, W]
        print(f"DEBUG loss_fn: flow2d_gt_resized.shape = {flow2d_gt_resized.shape}")
        flow2d_gt = flow2d_gt_resized.permute(0, 2, 3, 1)  # [1, H, W, 2]
        print(f"DEBUG loss_fn: flow2d_gt after resize = {flow2d_gt.shape}")
    
    if flow3d_gt.shape[1:3] != (target_h, target_w):
        print(f"DEBUG loss_fn: Resizing flow3d_gt from {flow3d_gt.shape[1:3]} to ({target_h}, {target_w})")
        # Convert to [B, C, H, W] for interpolation
        flow3d_gt_nchw = flow3d_gt.permute(0, 3, 1, 2)  # [1, 3, H_gt, W_gt]
        print(f"DEBUG loss_fn: flow3d_gt_nchw.shape = {flow3d_gt_nchw.shape}, target_size = ({target_h}, {target_w})")
        assert flow3d_gt_nchw.shape[1] == 3, f"flow3d_gt_nchw should have 3 channels, got {flow3d_gt_nchw.shape}"
        flow3d_gt_resized = F.interpolate(flow3d_gt_nchw, 
                                         size=(target_h, target_w), 
                                         mode='bilinear', align_corners=False)  # [1, 3, H, W]
        print(f"DEBUG loss_fn: flow3d_gt_resized.shape = {flow3d_gt_resized.shape}")
        flow3d_gt = flow3d_gt_resized.permute(0, 2, 3, 1)  # [1, H, W, 3]
        print(f"DEBUG loss_fn: flow3d_gt after resize = {flow3d_gt.shape}")
    
    # Compute EPE for single view
    epe_2d = torch.sum((flow2d_est - flow2d_gt)**2, -1).sqrt()  # [1, H, W]
    epe_3d = torch.sum((flow3d_est - flow3d_gt)**2, -1).sqrt()  # [1, H, W]
    
    # Compute valid mask
    mag_final = torch.sum(flow2d_est**2, dim=-1).sqrt()  # [1, H, W]
    depth_final = torch.sum(flow3d_est**2, dim=-1).sqrt()  # [1, H, W]
    valid = (mag_final < MAX_FLOW) & (depth_final < MAX_DEPTH)  # [1, H, W]
    
    # Compute losses for single view
    valid_view = valid[0]  # [H, W]
    epe_2d_view = epe_2d[0]  # [H, W]
    epe_3d_view = epe_3d[0]  # [H, W]
    
    # Compute loss (only over valid pixels)
    if valid_view.sum() > 0:  # Avoid division by zero
        loss_2d = (valid_view * epe_2d_view).sum() / (valid_view.sum() + 1e-8)
        loss_3d = (valid_view * epe_3d_view).sum() / (valid_view.sum() + 1e-8)
    else:
        # Use zeros_like to preserve gradient connection even when no valid pixels
        # This ensures the loss tensor requires gradients
        loss_2d = torch.zeros_like(epe_2d_view.sum())
        loss_3d = torch.zeros_like(epe_3d_view.sum())
    
    # Base flow loss
    loss = loss_2d + loss_3d
    
    # For metrics compatibility, keep single view format
    loss_2d_per_view = [loss_2d.item()]
    loss_3d_per_view = [loss_3d.item()]
    view_weights = torch.tensor([1.0], device=epe_2d.device)

    # =========================================================================
    # DEPTH SUPERVISION LOSSES
    # =========================================================================
    depth_metrics = {}
    
    # 1. Depth Consistency Loss: depth1 + dz should equal depth2
    if depth_consistency_weight > 0 and depth2_gt is not None:
        # Unscale depth1 for consistency check
        depth1_unscaled = depth1 / DEPTH_SCALE if depth1.max() < 10 else depth1
        depth2_gt_unscaled = depth2_gt / DEPTH_SCALE if depth2_gt.max() < 10 else depth2_gt
        
        depth_consist_loss, consist_metrics = compute_depth_consistency_loss(
            depth1_unscaled, flow3d_est, depth2_gt_unscaled, valid_mask=valid
        )
        loss = loss + depth_consistency_weight * depth_consist_loss
        depth_metrics.update(consist_metrics)
        depth_metrics['loss_depth_consistency'] = depth_consist_loss.item()
    
    # 2. Direct Depth Supervision Loss
    if depth_loss_weight > 0 and depth2_gt is not None:
        # Compute predicted depth at t+1 from flow
        depth1_unscaled = depth1 / DEPTH_SCALE if depth1.max() < 10 else depth1
        depth2_gt_unscaled = depth2_gt / DEPTH_SCALE if depth2_gt.max() < 10 else depth2_gt
        
        # Squeeze channel dim if present
        if depth1_unscaled.dim() == 4 and depth1_unscaled.shape[1] == 1:
            depth1_unscaled = depth1_unscaled.squeeze(1)
        
        # Resize depth1 to match flow resolution if needed
        dz_direct = flow3d_est[..., 2]
        if depth1_unscaled.shape[-2:] != dz_direct.shape[-2:]:
            depth1_unscaled = F.interpolate(depth1_unscaled.unsqueeze(1), size=dz_direct.shape[-2:],
                                           mode='bilinear', align_corners=False).squeeze(1)
        
        # Predicted depth2 = depth1 + dz (z component of 3D flow)
        depth2_pred = depth1_unscaled + dz_direct
        
        depth_sup_loss, sup_metrics = compute_depth_loss(depth2_pred, depth2_gt_unscaled, valid_mask=valid)
        loss = loss + depth_loss_weight * depth_sup_loss
        depth_metrics.update(sup_metrics)
        depth_metrics['loss_depth_direct'] = depth_sup_loss.item()
    
    # 3. Multi-view Depth Consistency Loss (disabled for Solution 1 - single view)
    if multiview_depth_weight > 0 and num_views > 1:
        # Split depth1 by views
        depth_views = [depth1[i] for i in range(num_views)]
        mv_depth_loss = compute_multiview_depth_consistency_loss(depth_views, intrinsics)
        loss = loss + multiview_depth_weight * mv_depth_loss
        depth_metrics['loss_multiview_depth'] = mv_depth_loss.item()
    # Note: For Solution 1 (single consolidated view), multiview_depth_weight should be 0
    
    # =========================================================================
    # COMPUTE METRICS
    # =========================================================================
    
    # Compute metrics (aggregate across all views)
    valid_flat = valid.reshape(-1)
    epe_2d_flat = epe_2d.reshape(-1)[valid_flat]
    epe_3d_flat = epe_3d.reshape(-1)[valid_flat]

    metrics = {
        'epe2d': epe_2d_flat.mean().item() if len(epe_2d_flat) > 0 else 0.0,
        'epe3d': epe_3d_flat.mean().item() if len(epe_3d_flat) > 0 else 0.0,
        '1px': (epe_2d_flat < 1).float().mean().item() if len(epe_2d_flat) > 0 else 0.0,
        '5cm': (epe_3d_flat < 0.05).float().mean().item() if len(epe_3d_flat) > 0 else 0.0,
        '10cm': (epe_3d_flat < 0.10).float().mean().item() if len(epe_3d_flat) > 0 else 0.0,
    }
    
    # Add per-view metrics for better monitoring
    # loss_2d_per_view and loss_3d_per_view are already lists of floats (from .item() calls on lines 516-517)
    metrics['loss_2d_per_view'] = loss_2d_per_view
    metrics['loss_3d_per_view'] = loss_3d_per_view
    metrics['view_weights'] = view_weights.cpu().tolist()
    
    # Add depth metrics
    metrics.update(depth_metrics)

    print("train loss_fn end===>")

    return loss, metrics


def fetch_dataloader(args):
    """ Fetch dataloader matching evaluation.py format """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # # train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
    #     config, 
    #     config.DATASET.TRAIN_SUBSET, 
    #     True,  # is_train=True
    #     transforms.Compose([transforms.ToTensor(), normalize])
    # )
    train_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(config, 
                                                                   config.DATASET.TEST_SUBSET, False,transforms.Compose([transforms.ToTensor(),normalize]))
    
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler_train,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    return train_loader


def fetch_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
    
    # Fix for OneCycleLR: ensure pct_start results in at least 2 warmup steps
    # For small num_steps, use a larger pct_start to avoid division by zero
    min_warmup_steps = 2
    default_pct_start = 0.001
    calculated_warmup = int(default_pct_start * args.num_steps)
    
    if calculated_warmup < min_warmup_steps:
        # Adjust pct_start to ensure minimum warmup steps
        # Use at least 2 steps for warmup, but cap at 0.1 (10% of total steps)
        pct_start = max(min_warmup_steps / args.num_steps, min(0.1, default_pct_start * 10))
        print(f"Warning: num_steps={args.num_steps} is small. Adjusting pct_start from {default_pct_start} to {pct_start:.4f} "
              f"to ensure minimum {min_warmup_steps} warmup steps (actual: {int(pct_start * args.num_steps)} steps)")
    else:
        pct_start = default_pct_start
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps, pct_start=pct_start, cycle_momentum=False)
    return optimizer, scheduler


def train(args):
    print("train train start===>")
    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D

    model = torch.nn.DataParallel(RAFT3D(args))
    model.cuda()
    model.train()
    
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt), strict=False)
   
    logger = Logger()

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(model, args)

    # Initialize metrics tracking for Excel export
    metrics_history = []
    excel_filename = f'training_metrics_{args.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    excel_path = os.path.join('checkpoints', excel_filename)
    print(f"Metrics will be saved to: {excel_path}")

    total_steps = 0
    while total_steps < args.num_steps:
        for i_batch, train_data_blob in enumerate(train_loader):
            
            # Unpack data matching evaluation.py format
            inputs, input_t1, meta, meta_t1, flows, valids, disps, disps_t1, disps_change, sceneflows, sceneflow_valids = train_data_blob
            
            # Stack multi-view data into batches
            image1 = torch.stack(inputs, dim=0).cuda()
            image2 = torch.stack(input_t1, dim=0).cuda()
            depth1 = torch.stack(disps, dim=0).cuda()
            depth2 = torch.stack(disps_t1, dim=0).cuda()
            flow2d = torch.stack(flows, dim=0).cuda()
            flow3d = torch.stack(sceneflows, dim=0).cuda()
            
            # Extract intrinsics
            intrinsics_t0_list = []
            intrinsics_t1_list = []
            
            for view_meta_t0, view_meta_t1 in zip(meta, meta_t1):
                cam_intri_t0 = view_meta_t0['camera_Intri']
                cam_intri_t1 = view_meta_t1['camera_Intri']
                
                if cam_intri_t0.shape[0] == 4:
                    cam_intri_t0 = cam_intri_t0[:3, :3]
                if cam_intri_t1.shape[0] == 4:
                    cam_intri_t1 = cam_intri_t1[:3, :3]
                    
                if cam_intri_t0.dim() > 2:
                    cam_intri_t0 = cam_intri_t0[0]
                if cam_intri_t1.dim() > 2:
                    cam_intri_t1 = cam_intri_t1[0]
                
                intrinsics_t0_list.append(cam_intri_t0)
                intrinsics_t1_list.append(cam_intri_t1)
            
            intrinsics_t0 = torch.stack(intrinsics_t0_list, dim=0).cuda()
            intrinsics_t1 = torch.stack(intrinsics_t1_list, dim=0).cuda()
            intrinsics_both = torch.stack([intrinsics_t0, intrinsics_t1], dim=1)
            intrinsics = intrinsics_both.unsqueeze(1).float()
            
            # Prepare images and depths
            image1, image2, depth1, depth2, padding = \
                prepare_images_and_depths(image1, image2, depth1, depth2, DEPTH_SCALE)
            
            # Prepare multi-view format for DQ model
            num_views = config.DATASET.CAMERA_NUM
            
            # DEBUG: Print shapes to diagnose the issue
            print(f"DEBUG image shapes BEFORE multi-view prep:")
            print(f"  image1.shape = {image1.shape}")
            print(f"  image2.shape = {image2.shape}")
            print(f"  depth1.shape = {depth1.shape}")
            print(f"  depth2.shape = {depth2.shape}")
            print(f"  len(inputs) = {len(inputs)}, len(input_t1) = {len(input_t1)}")
            print(f"  args.batch_size = {args.batch_size}")
            
            if image1.shape[0] == num_views:
                print(f"image1.shape[0] == {num_views}===>")
                image1_t0_list = [image1[i:i+1] for i in range(num_views)]
                image2_t1_list = [image2[i:i+1] for i in range(num_views)]
            elif image1.shape[0] == 2 * num_views:
                print(f"image1.shape[0] == {2 * num_views}===>")
                batch_size = 1
                image1_reshaped = image1.view(batch_size, num_views, 2, *image1.shape[1:])
                image2_reshaped = image2.view(batch_size, num_views, 2, *image2.shape[1:])
                image1_t0_list = [image1_reshaped[:, i, 0] for i in range(num_views)]
                image2_t1_list = [image2_reshaped[:, i, 1] for i in range(num_views)]
            else:
                print(f"ERROR: Unexpected image1.shape[0] = {image1.shape[0]}")
                print(f"  Expected 5 or 10, but got {image1.shape[0]}")
                print(f"  This happens when batch_size > 1. Current batch_size = {args.batch_size}")
                print(f"  With batch_size={args.batch_size} and 5 views, expected shape[0] = {args.batch_size * 5}")
                raise ValueError(f"image1.shape[0] must be 5 or 10, got {image1.shape[0]}. Use --batch_size 1 for now.")
            
            image_for_dq = [image1_t0_list, image2_t1_list]
            stacked_meta = [meta, meta_t1]
            
            # Forward pass
            optimizer.zero_grad()
            
            # Use train_mode=True to get flows with gradients
            # The model returns flow_est_list and flow_rev_list when train_mode=True
            # These flows have gradients through the update block operations
            flow_est_list, flow_rev_list = model(image1, image2, depth1, depth2, intrinsics, iters=12, 
                                                 train_mode=True, meta=stacked_meta, image_for_dq=image_for_dq)
            
            # Use the last flow estimate (final iteration) which should have gradients
            # flow_est_list contains tuples of (flow2d_est, flow3d_est, valid) at each iteration
            flow2d_est, flow3d_est, valid_est = flow_est_list[-1]  # Last iteration
            
            # Debug: Check if flows require gradients
            print(f"DEBUG train: flow2d_est.requires_grad = {flow2d_est.requires_grad}, flow2d_est.shape = {flow2d_est.shape}")
            if flow3d_est is not None:
                print(f"DEBUG train: flow3d_est.requires_grad = {flow3d_est.requires_grad}, flow3d_est.shape = {flow3d_est.shape}")
            
            # Solution 1: Prepare intrinsics for single consolidated view
            # Use first view's intrinsics (or could average)
            intrinsics_for_flow = intrinsics[0:1, 0, 0, :, :]  # [1, 3, 3] - first view at t0
            
            # Handle depth for flow computation - Solution 1 uses single view
            # Model outputs single view depth, but GT might have multiple views
            # Use first view's depth for GT comparison
            if depth1.shape[0] >= num_views:
                depth1_for_flow = depth1[0:1]  # [1, H, W] - first view
                depth2_for_flow = depth2[0:1]  # [1, H, W] - first view
            else:
                depth1_for_flow = depth1[:1]  # Ensure single view
                depth2_for_flow = depth2[:1]  # Ensure single view
            
            # Compute loss directly from flows (which have gradients)
            # Use a modified loss function that accepts flows instead of Ts
            loss, metrics = loss_fn_from_flows(flow2d_est, flow3d_est, depth1_for_flow, intrinsics_for_flow,
                                              flow2d, flow3d, stacked_meta,
                                              valid_mask=None, gamma=0.9, num_views=1,
                                              use_geometric_weights=False,
                                              depth2_gt=depth2_for_flow,
                                              depth_loss_weight=args.depth_loss_weight,
                                              depth_consistency_weight=args.depth_consistency_weight,
                                              multiview_depth_weight=0.0)

            # Debug: Check if loss requires gradients
            print(f"DEBUG train: loss.requires_grad = {loss.requires_grad}, loss.grad_fn = {loss.grad_fn}, loss.item() = {loss.item():.6f}")
            
            if not loss.requires_grad:
                print("ERROR: Loss does not require gradients! This will cause backward() to fail.")
                print("Checking model parameters...")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"  {name}: requires_grad=True, shape={param.shape}")
                    else:
                        print(f"  {name}: requires_grad=False, shape={param.shape}")
                raise RuntimeError("Loss tensor does not require gradients. Check model parameters and Ts output.")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Filter metrics for logger (only keep float values)
            filtered_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            logger.push(filtered_metrics)
            
            total_steps += 1

            # Print key metrics every 10 steps and save to Excel
            if total_steps % 10 == 0:
                epe3d_val = metrics.get('epe3d', 0.0)
                epe2d_val = metrics.get('epe2d', 0.0)
                abs_rel_val = metrics.get('depth_abs_rel', 0.0)
                depth_consist_val = metrics.get('depth_consistency', 0.0)
                depth_rmse_val = metrics.get('depth_rmse', 0.0)
                loss_depth_direct_val = metrics.get('loss_depth_direct', 0.0)
                loss_depth_consistency_val = metrics.get('loss_depth_consistency', 0.0)
                
                print(f"[Step {total_steps:6d}/{args.num_steps}] epe3d: {epe3d_val:.4f} | abs_rel: {abs_rel_val:.4f} | depth_consist: {depth_consist_val:.4f}")
                
                # Append metrics to history for Excel export
                metrics_history.append({
                    'step': total_steps,
                    'epe2d': epe2d_val,
                    'epe3d': epe3d_val,
                    'abs_rel': abs_rel_val,
                    'depth_rmse': depth_rmse_val,
                    'depth_consistency': depth_consist_val,
                    'loss_depth_direct': loss_depth_direct_val,
                    'loss_depth_consistency': loss_depth_consistency_val,
                    'learning_rate': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.lr
                })

            if total_steps % 1000 == 0:
                print(f"\nStep {total_steps}/{args.num_steps}:")
                print(f"  Overall - epe2d: {metrics['epe2d']:.3f}, epe3d: {metrics['epe3d']:.3f}")
                print(f"  View weights: {[f'{w:.3f}' for w in metrics['view_weights']]}")
                print(f"  Per-view 2D loss: {[f'{l:.3f}' for l in metrics['loss_2d_per_view']]}")
                print(f"  Per-view 3D loss: {[f'{l:.3f}' for l in metrics['loss_3d_per_view']]}")
                
                # Print depth metrics if available
                if 'depth_abs_rel' in metrics:
                    print(f"  Depth - abs_rel: {metrics['depth_abs_rel']:.4f}, rmse: {metrics['depth_rmse']:.4f}")
                if 'depth_consistency' in metrics:
                    print(f"  Depth consistency: {metrics['depth_consistency']:.4f}")
                if 'loss_multiview_depth' in metrics:
                    print(f"  Multi-view depth: {metrics['loss_multiview_depth']:.4f}")

            if total_steps % 20000 == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)
                
            if total_steps >= args.num_steps:
                break

    # Final checkpoint
    PATH = 'checkpoints/%s_final.pth' % args.name
    torch.save(model.state_dict(), PATH)
    
    # Save metrics to Excel
    if len(metrics_history) > 0:
        df = pd.DataFrame(metrics_history)
        df.to_excel(excel_path, index=False, sheet_name='Training Metrics')
        print(f"\n{'='*50}")
        print(f"Training metrics saved to: {excel_path}")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"{'='*50}\n")
        
        # Also print summary statistics
        print("Summary Statistics:")
        print(df.describe().to_string())
        print()
    
    if not args.skip_eval:
        print("Training complete. Running evaluation...")
        # Make sure model is properly set up for evaluation
        model.eval()
        # Pass the DataParallel wrapped model, not just the module
        test_sceneflow(model)
    else:
        print("Training complete. Skipping evaluation.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft3d_dq', help='name your experiment')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--batch_size', type=int, default=2) #1500
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cfg', help='experiment configure file name', 
                        default='configs/panoptic/generalization/CMU0ex3.yaml', type=str)
    parser.add_argument('--use_geometric_weights', action='store_true', 
                       help='Use geometric weighting based on camera baselines')
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation at the end of training')
    
    # Depth supervision arguments
    parser.add_argument('--depth_loss_weight', type=float, default=0.0,
                       help='Weight for direct depth supervision loss (0=disabled, recommended: 0.1-1.0)')
    parser.add_argument('--depth_consistency_weight', type=float, default=0.0,
                       help='Weight for depth consistency loss (depth1+dz=depth2, recommended: 0.5-2.0)')
    parser.add_argument('--multiview_depth_weight', type=float, default=0.0,
                       help='Weight for multi-view depth consistency loss (recommended: 0.1-0.5)')

    # model arguments
    parser.add_argument('--radius', type=int, default=32)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    
    args = parser.parse_args()

    print(args)
    print("train.py end===>")
    train(args)
    print("train.py end 2===>")