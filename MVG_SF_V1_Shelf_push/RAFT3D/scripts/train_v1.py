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
import raft3d.projective_ops as pops

from utils import Logger, show_image, normalize_image
from evaluation import test_sceneflow
from data_readers.sceneflow import SceneFlow

import torchvision.transforms as transforms
from lib.core.config import config
from lib.core.config import update_config, update_config_dynamic_input
import lib.dataset as dataset


RV_WEIGHT = 0.2
DZ_WEIGHT = 100.0
DEPTH_SCALE = 0.2
MAX_DEPTH = 250
MAX_FLOW = 250


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
    
    return torch.tensor(weights, dtype=torch.float32)


def loss_fn(Ts, depth1, intrinsics, flow2d_gt, flow3d_gt, meta, valid_mask=None, 
            gamma=0.9, num_views=5, use_geometric_weights=True):
    """ 
    Loss function using transformation field predictions with multi-view support.
    
    Args:
        Ts: SE3 transformation field
        depth1: Depth at time t
        intrinsics: Camera intrinsic parameters
        flow2d_gt: Ground truth 2D optical flow
        flow3d_gt: Ground truth 3D scene flow
        meta: List of metadata dicts (for computing geometric weights)
        valid_mask: Optional pre-computed valid mask
        gamma: Decay factor (unused, for compatibility)
        num_views: Number of camera views
        use_geometric_weights: If True, use baseline-based weighting; else equal weights
    
    Returns:
        loss: Total loss value
        metrics: Dictionary of evaluation metrics
    """
    
    # Extract flow from transformation field
    flow2d_est, flow3d_est, _ = pops.induced_flow(Ts, depth1, intrinsics)
    
    # Unpad and unscale
    flow2d_est = flow2d_est[:, :-4, :, :2]  # Remove padding
    flow3d_est = flow3d_est[:, :-4] / DEPTH_SCALE  # Undo depth scaling
    
    # Resize ground truth to match estimated flow resolution if needed
    if flow2d_gt.shape[1:3] != flow2d_est.shape[1:3]:
        if len(flow2d_gt.shape) == 5:  # Multi-view format [V, B, C, H, W]
            flow2d_gt = flow2d_gt.squeeze(1)  # [V, C, H, W]
            flow2d_gt = F.interpolate(flow2d_gt, 
                                     size=flow2d_est.shape[1:3], 
                                     mode='bilinear', align_corners=False)
            flow2d_gt = flow2d_gt.permute(0, 2, 3, 1)  # [V, H, W, C]
        else:
            flow2d_gt = F.interpolate(flow2d_gt.permute(0, 3, 1, 2), 
                                     size=flow2d_est.shape[1:3], 
                                     mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    
    if flow3d_gt.shape[1:3] != flow3d_est.shape[1:3]:
        if len(flow3d_gt.shape) == 5:  # Multi-view format [V, B, C, H, W]
            flow3d_gt = flow3d_gt.squeeze(1)  # [V, C, H, W]
            flow3d_gt = F.interpolate(flow3d_gt, 
                                     size=flow3d_est.shape[1:3], 
                                     mode='bilinear', align_corners=False)
            flow3d_gt = flow3d_gt.permute(0, 2, 3, 1)  # [V, H, W, C]
        else:
            flow3d_gt = F.interpolate(flow3d_gt.permute(0, 3, 1, 2), 
                                     size=flow3d_est.shape[1:3], 
                                     mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    
    # Compute EPE per view
    epe_2d = torch.sum((flow2d_est - flow2d_gt)**2, -1).sqrt()  # [num_views, H, W]
    epe_3d = torch.sum((flow3d_est - flow3d_gt)**2, -1).sqrt()  # [num_views, H, W]
    
    # Compute valid mask per view
    mag_final = torch.sum(flow2d_est**2, dim=-1).sqrt()  # [num_views, H, W]
    depth_final = torch.sum(flow3d_est**2, dim=-1).sqrt()  # [num_views, H, W]
    valid = (mag_final < MAX_FLOW) & (depth_final < MAX_DEPTH)  # [num_views, H, W]
    
    # Compute per-view losses and then weighted average across views
    loss_2d_per_view = []
    loss_3d_per_view = []
    
    for view_idx in range(num_views):
        valid_view = valid[view_idx]  # [H, W]
        epe_2d_view = epe_2d[view_idx]  # [H, W]
        epe_3d_view = epe_3d[view_idx]  # [H, W]
        
        # Compute loss for this view (only over valid pixels)
        if valid_view.sum() > 0:  # Avoid division by zero
            loss_2d_view = (valid_view * epe_2d_view).sum() / (valid_view.sum() + 1e-8)
            loss_3d_view = (valid_view * epe_3d_view).sum() / (valid_view.sum() + 1e-8)
        else:
            loss_2d_view = torch.tensor(0.0, device=epe_2d.device)
            loss_3d_view = torch.tensor(0.0, device=epe_3d.device)
        
        loss_2d_per_view.append(loss_2d_view)
        loss_3d_per_view.append(loss_3d_view)
    
    # Compute view weights based on camera geometry
    if use_geometric_weights and meta is not None:
        try:
            view_weights = compute_geometric_weights(meta[0], num_views)  # meta[0] is t0 metadata
            view_weights = view_weights.to(epe_2d.device)
        except Exception as e:
            print(f"Warning: Failed to compute geometric weights: {e}")
            print("Falling back to equal weights")
            view_weights = torch.ones(num_views, device=epe_2d.device) / num_views
    else:
        # Equal weighting (original approach)
        view_weights = torch.ones(num_views, device=epe_2d.device) / num_views
    
    # Weighted average losses across all views
    loss_2d = sum(w * l for w, l in zip(view_weights, loss_2d_per_view))
    loss_3d = sum(w * l for w, l in zip(view_weights, loss_3d_per_view))
    
    loss = loss_2d + loss_3d

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
    metrics['loss_2d_per_view'] = [l.item() for l in loss_2d_per_view]
    metrics['loss_3d_per_view'] = [l.item() for l in loss_3d_per_view]
    metrics['view_weights'] = view_weights.cpu().tolist()

    return loss, metrics


def fetch_dataloader(args):
    """ Fetch dataloader matching evaluation.py format """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, 
        config.DATASET.TRAIN_SUBSET, 
        True,  # is_train=True
        transforms.Compose([transforms.ToTensor(), normalize])
    )
    
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
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps, pct_start=0.001, cycle_momentum=False)
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
            num_views = 5
            if image1.shape[0] == 5:
                image1_t0_list = [image1[i:i+1] for i in range(5)]
                image2_t1_list = [image2[i:i+1] for i in range(5)]
            elif image1.shape[0] == 10:
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
            
            # Prepare intrinsics for flow computation
            intrinsics_for_flow = intrinsics[:, 0, 0, :, :]
            
            # Handle depth for flow computation
            if depth1.shape[0] == 10:
                depth1_for_flow = depth1[:5]
            else:
                depth1_for_flow = depth1
            
            # Compute loss
            loss, metrics = loss_fn(Ts, depth1_for_flow, intrinsics_for_flow, 
                                   flow2d, flow3d, stacked_meta, 
                                   valid_mask=None, gamma=0.9, num_views=5,
                                   use_geometric_weights=args.use_geometric_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            logger.push(metrics)
            
            total_steps += 1

            if total_steps % 1000 == 0:
                print(f"\nStep {total_steps}/{args.num_steps}:")
                print(f"  Overall - epe2d: {metrics['epe2d']:.3f}, epe3d: {metrics['epe3d']:.3f}")
                print(f"  View weights: {[f'{w:.3f}' for w in metrics['view_weights']]}")
                print(f"  Per-view 2D loss: {[f'{l:.3f}' for l in metrics['loss_2d_per_view']]}")
                print(f"  Per-view 3D loss: {[f'{l:.3f}' for l in metrics['loss_3d_per_view']]}")

            if total_steps % 20000 == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)
                
            if total_steps >= args.num_steps:
                break

    # Final checkpoint
    PATH = 'checkpoints/%s_final.pth' % args.name
    torch.save(model.state_dict(), PATH)
    
    print("Training complete. Running evaluation...")
    model.eval()
    test_sceneflow(model.module)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft3d_dq', help='name your experiment')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_geometric_weights', action='store_true', 
                       help='Use geometric weighting based on camera baselines')

    # model arguments
    parser.add_argument('--radius', type=int, default=32)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    
    args = parser.parse_args()

    print(args)
    train(args)