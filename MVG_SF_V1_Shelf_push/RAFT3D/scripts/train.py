import sys
sys.path.append('.')

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


RV_WEIGHT = 0.2  # Reverse flow weight for training stability
DEPTH_SCALE = 0.05 #0.2
MAX_DEPTH = 150 #250
MAX_FLOW = 150 #250


def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=0.2):
    """ padding, normalization, and scaling """
    
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

    return image1, image2, depth1, depth2, (pad_w, pad_h)



def compute_reverse_flow(Ts, depth1, intrinsics):
    """
     Compute reverse flow for training stability (from train_things.py).
    Reverse flow helps regularize the SE(3) field by enforcing cycle consistency.
    
    Args:
        Ts: Forward SE(3) transformation field
        depth1: Depth at time t
        intrinsics: Camera intrinsics
    
    Returns:
        flow2d_reverse: Reverse 2D flow [1, H, W, 2]
    """
    # Invert the transformation field
    Ts_inv = Ts.inv()
    
    # Extract reverse flow
    flow2d_rev, _, _ = pops.induced_flow(Ts_inv, depth1, intrinsics)
    
    return flow2d_rev


def loss_fn(Ts, depth1, intrinsics, flow2d_gt, flow3d_gt, valid_mask=None, gamma=0.9, use_rv_loss=True):
    """ 
    Loss function for MULTI-VIEW scene flow (matching evaluation.py format).
    
    NOTE: This matches evaluation.py expectations:
    - Model outputs aggregated Ts [1, H, W], which is replicated to [num_views, H, W]
    - Flow is computed per-view using multi-view depth and intrinsics
    - Loss is computed per-view and averaged (not aggregated before loss)
    
    Args:
        Ts: SE3 transformation field [num_views, H, W] (replicated from aggregated model output)
        depth1: Depth at time t [num_views, 1, H, W] or [num_views, H, W]
        intrinsics: Camera intrinsic parameters [num_views, 3, 3] (all views, matching evaluation.py)
        flow2d_gt: Ground truth 2D optical flow [num_views, H, W, 2] or [num_views, 1, 2, H, W]
        flow3d_gt: Ground truth 3D scene flow [num_views, H, W, 3] or [num_views, 1, 3, H, W]
        valid_mask: Optional pre-computed valid mask
        gamma: Decay factor for iterative refinement (if applicable)
        use_rv_loss: If True, add reverse flow loss for training stability
    
    Returns:
        loss: Total loss value (averaged across all valid pixels across all views)
        metrics: Dictionary of evaluation metrics
    """
    
    # Upsample Ts to match depth1 resolution if needed (multi-view format like evaluation.py)
    # Determine target resolution from depth1
    if depth1.dim() == 4:  # [num_views, channels, H, W]
        target_h, target_w = depth1.shape[2], depth1.shape[3]
    elif depth1.dim() == 3:  # [num_views, H, W]
        target_h, target_w = depth1.shape[1], depth1.shape[2]
    else:
        target_h, target_w = depth1.shape[-2], depth1.shape[-1]
    
    # Upsample Ts if needed (same logic as evaluation.py lines 348-360)
    if Ts.shape[1] != target_h or Ts.shape[2] != target_w:
        # Ts.data has shape [num_views, H, W, 7] - need to permute for interpolation
        Ts_data = Ts.data  # [num_views, H_low, W_low, 7]
        Ts_data = Ts_data.permute(0, 3, 1, 2)  # [num_views, 7, H_low, W_low]
        Ts_data_upsampled = F.interpolate(Ts_data, size=(target_h, target_w), mode='bilinear', align_corners=False)
        Ts_data_upsampled = Ts_data_upsampled.permute(0, 2, 3, 1)  # [num_views, H, W, 7]
        
        # Create new SE3 object with upsampled data (preserves gradients)
        Ts = SE3(Ts_data_upsampled)
    
    # Extract flow from transformation field (multi-view format)
    # Ts is [num_views, H, W], depth1 is [num_views, 1, H, W] or [num_views, H, W]
    # intrinsics is [num_views, 4] or [num_views, 3, 3]
    flow2d_est, flow3d_est, _ = pops.induced_flow(Ts, depth1, intrinsics)
    
    # Unpad and unscale (multi-view format)
    flow2d_est = flow2d_est[:, :-4, :, :2]  # Remove padding [num_views, H-4, W, 2]
    flow3d_est = flow3d_est[:, :-4] / DEPTH_SCALE  # Undo depth scaling [num_views, H-4, W, 3]
    
    # Handle ground truth dimensions - following evaluation.py pattern
    # Reshape ground truth to match estimated flow format (same as evaluation.py)
    
    # Handle flow2d_gt: Check for 5D format [num_views, 2, 2, H, W] or [num_views, 2, H, W]
    if flow2d_gt.dim() == 5 and flow2d_gt.shape[1] == 2 and flow2d_gt.shape[2] == 2:
        # [num_views, 2, 2, H, W] -> [num_views, 2, H, W] -> [num_views, H, W, 2]
        flow2d_gt = flow2d_gt[:, 0, :, :, :].permute(0, 2, 3, 1)  # [num_views, H, W, 2]
    
    # Handle flow3d_gt: Check for 5D format [num_views, 2, 3, H, W]
    if flow3d_gt.dim() == 5 and flow3d_gt.shape[1] == 2:
        # [num_views, 2, 3, H, W] -> [num_views, 3, H, W] -> [num_views, H, W, 3]
        flow3d_gt = flow3d_gt[:, 0, :, :, :].permute(0, 2, 3, 1)  # [num_views, H, W, 3]
    
    # Handle 5D format [num_views, 1, C, H, W] (similar to evaluation.py)
    if flow2d_gt.dim() == 5:  # Multi-view format [num_views, 1, 2, H, W]
        # Reshape to [num_views, 2, H, W] for interpolation
        flow2d_gt = flow2d_gt.squeeze(1)  # Remove singleton dimension -> [num_views, 2, H, W]
        if flow2d_gt.shape[2:4] != flow2d_est.shape[1:3]:
            flow2d_gt = F.interpolate(flow2d_gt, 
                                     size=flow2d_est.shape[1:3], 
                                     mode='bilinear', align_corners=False)
        # Convert to [num_views, H, W, 2] to match flow2d_est format
        flow2d_gt = flow2d_gt.permute(0, 2, 3, 1)
    elif flow2d_gt.dim() == 4:
        # Check if it's [num_views, C, H, W] or [num_views, H, W, C]
        if flow2d_gt.shape[-1] == 2 or flow2d_gt.shape[-1] == 3:
            # Format is [num_views, H, W, C] - standard format
            if flow2d_gt.shape[1:3] != flow2d_est.shape[1:3]:
                flow2d_gt = F.interpolate(flow2d_gt.permute(0, 3, 1, 2), 
                                         size=flow2d_est.shape[1:3], 
                                         mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        else:
            # Format is [num_views, C, H, W] - need to permute first
            if flow2d_gt.shape[2:4] != flow2d_est.shape[1:3]:
                flow2d_gt = F.interpolate(flow2d_gt, 
                                         size=flow2d_est.shape[1:3], 
                                         mode='bilinear', align_corners=False)
            flow2d_gt = flow2d_gt.permute(0, 2, 3, 1)  # [num_views, H, W, C]
    
    # Handle flow3d_gt: 5D format [num_views, 1, 3, H, W]
    if flow3d_gt.dim() == 5:  # Multi-view format [num_views, 1, 3, H, W]
        # Reshape to [num_views, 3, H, W] for interpolation
        flow3d_gt = flow3d_gt.squeeze(1)  # Remove singleton dimension -> [num_views, 3, H, W]
        if flow3d_gt.shape[2:4] != flow3d_est.shape[1:3]:
            flow3d_gt = F.interpolate(flow3d_gt, 
                                     size=flow3d_est.shape[1:3], 
                                     mode='bilinear', align_corners=False)
        # Convert to [num_views, H, W, 3] to match flow3d_est format
        flow3d_gt = flow3d_gt.permute(0, 2, 3, 1)
    elif flow3d_gt.dim() == 4:
        # Check if it's [num_views, C, H, W] or [num_views, H, W, C]
        if flow3d_gt.shape[-1] == 3:
            # Format is [num_views, H, W, C] - standard format
            if flow3d_gt.shape[1:3] != flow3d_est.shape[1:3]:
                flow3d_gt = F.interpolate(flow3d_gt.permute(0, 3, 1, 2), 
                                         size=flow3d_est.shape[1:3], 
                                         mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        else:
            # Format is [num_views, C, H, W] - need to permute first
            if flow3d_gt.shape[2:4] != flow3d_est.shape[1:3]:
                flow3d_gt = F.interpolate(flow3d_gt, 
                                         size=flow3d_est.shape[1:3], 
                                         mode='bilinear', align_corners=False)
            flow3d_gt = flow3d_gt.permute(0, 2, 3, 1)  # [num_views, H, W, C]
    
    # Keep multi-view format (do NOT aggregate) - matching evaluation.py
    # flow2d_est and flow3d_est are [num_views, H, W, C]
    # flow2d_gt and flow3d_gt should be [num_views, H, W, C] after reshaping
    
    # Compute EPE (End-Point Error) per-view
    epe_2d = torch.sum((flow2d_est - flow2d_gt)**2, -1).sqrt()  # [num_views, H, W]
    # print("1_epe_2d===>" , epe_2d)
    # print("flow2d_est===>" , flow2d_est)
    # print("flow2d_gt===>" , flow2d_gt)      
    epe_3d = torch.sum((flow3d_est - flow3d_gt)**2, -1).sqrt()  # [num_views, H, W]
    print("flow3d_est===>" , flow3d_est)
    print("flow3d_gt===>" , flow3d_gt)
    print("1_epe_3d===>" , epe_3d)
    
    # Compute valid mask per-view (matching evaluation.py)
    if valid_mask is None:
        mag_final = torch.sum(flow2d_est**2, dim=-1).sqrt()  # [num_views, H, W]
        print("mag_final===>" , mag_final)
        depth_final = torch.sum(flow3d_est**2, dim=-1).sqrt()  # [num_views, H, W]
        print("depth_final===>" , depth_final)
        valid = (mag_final < MAX_FLOW) & (depth_final < MAX_DEPTH)  # [num_views, H, W]
        
        # Diagnostic logging when valid mask is empty
        if valid.sum() == 0:
            mag_max = mag_final.max().item()
            mag_mean = mag_final.mean().item()
            depth_max = depth_final.max().item()
            depth_mean = depth_final.mean().item()
            print(f"\n WARNING: valid.sum() == 0 (no valid pixels)")
            print(f"  - 2D flow magnitude: max={mag_max:.2f}, mean={mag_mean:.2f} (threshold={MAX_FLOW})")
            print(f"  - 3D flow magnitude: max={depth_max:.2f}, mean={depth_mean:.2f} (threshold={MAX_DEPTH})")
            print(f"  - This usually indicates model predictions are too large (scale mismatch or untrained model)")
            print(f"\n   DEPTH_SCALE Diagnostic:")
            print(f"     Current DEPTH_SCALE = {DEPTH_SCALE}")
            print(f"     If 3D flow is too large, try DECREASING DEPTH_SCALE (e.g., 0.1 or 0.15)")
            print(f"     If 3D flow is too small, try INCREASING DEPTH_SCALE (e.g., 0.3 or 0.4)")
            print(f"     Rule of thumb: DEPTH_SCALE should make scaled depth values in range [0.5, 50.0]")
            print(f"     Check: model initialization, depth scaling, and intrinsics scaling")
    else:
        valid = valid_mask
    
    # Compute losses per-view and average (matching evaluation.py multi-view format)
    # Always compute using operations that maintain gradients, even when valid.sum() == 0
    loss_2d = (valid * epe_2d).sum() / (valid.sum() + 1e-8)
    loss_3d = (valid * epe_3d).sum() / (valid.sum() + 1e-8)
    print("loss_2d===>" , loss_2d)
    print("loss_3d===>" , loss_3d)
    
    # Add reverse flow loss for training stability (from train_things.py)
    # Initialize with zero that maintains gradient connection
    loss_rv = epe_2d.sum() * 0.0  # Maintains gradient connection
    if use_rv_loss:
        flow2d_rev = compute_reverse_flow(Ts, depth1, intrinsics)
        flow2d_rev = flow2d_rev[:, :-4, :, :2]  # Remove padding to match dimensions [num_views, H-4, W, 2]
        
        # Compute reverse flow error per-view
        epe_2d_rev = torch.sum((flow2d_rev - flow2d_gt)**2, -1).sqrt()  # [num_views, H, W]
        
        # Always compute using operations that maintain gradients
        loss_rv = RV_WEIGHT * (valid * epe_2d_rev).sum() / (valid.sum() + 1e-8)
    
    # Total loss (averaged across all valid pixels across all views)
    # loss = loss_2d + loss_3d + loss_rv
    # loss =  loss_2d + loss_3d + loss_rv
    loss = loss_3d

    # Compute metrics (flatten across all views, matching evaluation.py)
    valid_flat = valid.reshape(-1)  # [num_views * H * W]

    # if valid.sum() == 0:
    #     # Use all pixels when no valid pixels (e.g., when GT is zeros)
    #     epe_2d_flat = epe_2d.reshape(-1)  # All pixels
    #     epe_3d_flat = epe_3d.reshape(-1)  # All pixels
    # else:
    #     # Use only valid pixels when available
    #     epe_2d_flat = epe_2d.reshape(-1)[valid_flat]  # Only valid pixels
    #     epe_3d_flat = epe_3d.reshape(-1)[valid_flat]  # Only valid pixels
    epe_2d_flat = epe_2d.reshape(-1)[valid_flat]  # Only valid pixels
    epe_3d_flat = epe_3d.reshape(-1)[valid_flat]  # Only valid pixels

    metrics = {
        'epe2d': epe_2d_flat.mean().item() if len(epe_2d_flat) > 0 else 0.0,
        'epe3d': epe_3d_flat.mean().item() if len(epe_3d_flat) > 0 else 0.0,
        '1px': (epe_2d_flat < 1).float().mean().item() if len(epe_2d_flat) > 0 else 0.0,
        '5cm': (epe_3d_flat < 0.05).float().mean().item() if len(epe_3d_flat) > 0 else 0.0,
        '10cm': (epe_3d_flat < 0.10).float().mean().item() if len(epe_3d_flat) > 0 else 0.0,
        'loss_2d': loss_2d.item(),
        'loss_3d': loss_3d.item(),
        'loss_rv': loss_rv.item() if use_rv_loss else 0.0,
    }

    return loss, metrics


def fetch_dataloader(args):
    """ Fetch dataloader matching evaluation.py format """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, 
        config.DATASET.TEST_SUBSET, 
        True,  # is_train=True
        transforms.Compose([transforms.ToTensor(), normalize]))

    from torch.utils.data import Subset

    total_frames = len(train_dataset)    
    print('total_frames===>', total_frames)      # total “timesteps”
    if total_frames > 0:
        end_idx    = int(0.7 * total_frames)    # 70% point ⇒ last 30%
        indices      = list(range( 0, end_idx))

        train_dataset_subset = Subset(train_dataset, indices)

        sampler_train = torch.utils.data.SequentialSampler(train_dataset_subset)
        train_loader = torch.utils.data.DataLoader(
                        train_dataset_subset,
                        batch_size=config.TRAIN.BATCH_SIZE,
                        sampler=sampler_train,
                        pin_memory=True,
                        num_workers=config.WORKERS)
    
    # sampler_train = torch.utils.data.RandomSampler(train_dataset)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     sampler=sampler_train,
    #     pin_memory=True,
    #     num_workers=args.num_workers,
    #     drop_last=True
    # )
    
    return train_loader


def fetch_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps, pct_start=0.02, cycle_momentum=False)
    return optimizer, scheduler


def train(args):

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

    total_steps = 0
    total_epe3d = 0.0
    total_1px = 0.0
    total_5cm = 0.0
    total_10cm = 0.0
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
            # Diagnostic: Check raw depth ranges before scaling
            if total_steps % 100 == 0:  # Print every 100 steps to avoid spam
                depth1_raw_min = depth1.min().item()
                depth1_raw_max = depth1.max().item()
                depth1_raw_mean = depth1.mean().item()
                print(f"\n[Step {total_steps}] Raw depth statistics (before scaling):")
                print(f"  depth1: min={depth1_raw_min:.3f}, max={depth1_raw_max:.3f}, mean={depth1_raw_mean:.3f}")
                print(f"  DEPTH_SCALE={DEPTH_SCALE} → scaled range: [{depth1_raw_min*DEPTH_SCALE:.3f}, {depth1_raw_max*DEPTH_SCALE:.3f}]")
            
            image1, image2, depth1, depth2, padding = \
                prepare_images_and_depths(image1, image2, depth1, depth2, DEPTH_SCALE)
            
            # Prepare multi-view format for DQ model
            num_views = config.DATASET.CAMERA_NUM
            if image1.shape[0] == num_views:
                image1_t0_list = [image1[i:i+1] for i in range(num_views)]
                image2_t1_list = [image2[i:i+1] for i in range(num_views)]
            elif image1.shape[0] == 2 * num_views:
                batch_size = 1
                image1_reshaped = image1.view(batch_size, num_views, 2, *image1.shape[1:])
                image2_reshaped = image2.view(batch_size, num_views, 2, *image2.shape[1:])
                image1_t0_list = [image1_reshaped[:, i, 0] for i in range(num_views)]
                image2_t1_list = [image2_reshaped[:, i, 1] for i in range(num_views)]
            
            image_for_dq = [image1_t0_list, image2_t1_list]
            stacked_meta = [meta, meta_t1]
            
            # Forward pass
            optimizer.zero_grad()
            
            Ts = model(image1, image2, depth1, depth2, intrinsics, iters=12, 
                      meta=stacked_meta, image_for_dq=image_for_dq)
            
            # Match evaluation.py multi-view format:
            # 1. Replicate aggregated Ts to match num_views (model returns [1, H, W], need [num_views, H, W])
            # 2. Use multi-view depth and intrinsics (not aggregated)
            
            # Get number of views
            num_views = config.DATASET.CAMERA_NUM
            
            # Replicate Ts to match multi-view format (same as evaluation.py expects)
            # Ts from model is [1, H, W] (aggregated), need [num_views, H, W]
            if Ts.shape[0] == 1:
                # Replicate Ts.data to match num_views
                Ts_data = Ts.data  # [1, H, W, 7]
                Ts_data_replicated = Ts_data.repeat(num_views, 1, 1, 1)  # [num_views, H, W, 7]
                Ts_multi = SE3(Ts_data_replicated)  # Create SE3 object with multi-view data
            else:
                Ts_multi = Ts
            
            # Prepare intrinsics for flow computation (multi-view format like evaluation.py)
            # Shape: [num_views, 1, 2, 3, 3] -> [num_views, 3, 3]
            # Use [num_views, 3, 3] format to match evaluation.py exactly (line 321)
            intrinsics_for_flow = intrinsics[:, 0, 0, :, :]  # [num_views, 3, 3] - all views, first batch, time t=0
            
            # Handle depth for flow computation (multi-view format like evaluation.py)
            # Use all views, not just first view
            if depth1.shape[0] == 2 * num_views:
                # If depth1 has 2*num_views (num_views × 2 timesteps), take first num_views
                depth1_for_flow = depth1[:num_views]  # [num_views, 1, H, W]
            elif depth1.shape[0] == num_views:
                depth1_for_flow = depth1  # [num_views, 1, H, W]
            else:
                # If single view, replicate to match num_views
                depth1_for_flow = depth1.repeat(num_views, 1, 1, 1)  # [num_views, 1, H, W]
            
            # Compute loss with reverse flow for stability (multi-view format)
            loss, metrics = loss_fn(Ts_multi, depth1_for_flow, intrinsics_for_flow, 
                                   flow2d, flow3d, 
                                   valid_mask=None, gamma=0.9, use_rv_loss=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            logger.push(metrics)
            # accumulate per-step metrics
            total_epe3d += metrics['epe3d']
            total_1px   += metrics['1px']
            total_5cm   += metrics['5cm']
            total_10cm  += metrics['10cm']
            
            total_steps += 1

            if total_steps % 1 == 0:
                print(f"\nStep {total_steps}/{args.num_steps}:")
                print(f"  Loss - 2D: {metrics['loss_2d']:.4f}, 3D: {metrics['loss_3d']:.4f}, RV: {metrics['loss_rv']:.4f}")
                print(f"  EPE - 2D: {metrics['epe2d']:.3f}, 3D: {metrics['epe3d']:.3f}")
                print(f"  Accuracy - 1px: {metrics['1px']:.3f}, 5cm: {metrics['5cm']:.3f}, 10cm: {metrics['10cm']:.3f}")

            if total_steps % 1000 == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)
                
            if total_steps >= args.num_steps:
                break

    # Final checkpoint
    PATH = 'checkpoints/%s_final.pth' % args.name
    torch.save(model.state_dict(), PATH)
    # compute overall training metrics
    avg_epe3d = total_epe3d / max(total_steps, 1)
    avg_1px   = total_1px   / max(total_steps, 1)
    avg_5cm   = total_5cm   / max(total_steps, 1)
    avg_10cm  = total_10cm  / max(total_steps, 1)
    print("=" *50)
    print("\nFinal training metrics over {} batches:".format(total_steps))
    print(f"  EPE3D: {avg_epe3d:.4f}")
    print(f"  1px accuracy:  {avg_1px:.4f}")
    print(f"  5cm accuracy:  {avg_5cm:.4f}")
    print(f" 10cm accuracy:  {avg_10cm:.4f}")
    print("=" *50)
    print("Training complete. Running evaluation...")
    model.eval()
    test_sceneflow(model.module)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft3d_mvgformer', help='name your experiment')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Batch size (recommend 1 for multi-view aggregated architecture)')
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cfg', help='experiment configure file name', 
                       default='/mnt/MIG_archive24/Datasets/iota/JagritiDatasets/MVG_SF_V1_Shelf/configs/shelf_campus/shelf_knn5-lr4-q1024.yaml', type=str)

    # model arguments
    parser.add_argument('--radius', type=int, default=32)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    
    args = parser.parse_args()
    
    # Load MVGFormer config
    update_config(args.cfg)
    print(f" --Loaded config from: {args.cfg}")
    print(f" --Number of cameras configured: {config.DATASET.CAMERA_NUM}")
    print(f" --Training configuration:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Total steps: {args.num_steps}")
    print(f"  - Workers: {args.num_workers}")

    print(args)
    train(args)