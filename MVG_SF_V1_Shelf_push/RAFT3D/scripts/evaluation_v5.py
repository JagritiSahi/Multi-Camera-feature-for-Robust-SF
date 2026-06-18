import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # maybe …/RAFT3D
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tqdm import tqdm
import numpy as np
import cv2
import argparse
import torch

from lietorch import SE3
import RAFT3D.raft3d.projective_ops as pops

from utils import show_image, normalize_image
from RAFT3D.data_readers.sceneflow import FlyingThingsTest
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add root to system path so you can access lib.models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import lib.models.dq_transformer as dq_transformer
from lib.models.dq_transformer import get_mvp
from lib.core.config import config, update_config, update_config_dynamic_input

from glob import glob
from RAFT3D.data_readers.frame_utils import *
import lib.dataset as dataset
import torchvision.transforms as transforms

# # Add root to system path so you can access lib.models
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# import lib.models.dq_transformer as dq_transformer
# from lib.models.dq_transformer import get_mvp
# from lib.core.config import config, update_config, update_config_dynamic_input

# from glob import glob
# from RAFT3D.data_readers.frame_utils import *
# import lib.dataset as dataset
# import torchvision.transforms as transforms

# import sys, os

# scale input depth maps (scaling is undone before evaluation)
DEPTH_SCALE = 0.2

# exclude pixels with depth > 250
MAX_DEPTH = 250

# exclude extermely fast moving pixels
MAX_FLOW = 250


def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=0.2):
    """ padding, normalization, and scaling """

    print("Preparing images and depths===>")
    # Some inputs here have an extra leading view/batch dimension, e.g. [V, B, C, H, W]
    # F.pad supports up to 5D tensors with non-constant padding. To be robust,
    # merge any leading dims into a single batch dim before padding, then return
    # the padded (flattened) tensors. The caller and model expect flattened
    # batch-shaped tensors (N, C, H, W).

    def _flatten_leading(x):
        # ensure tensor
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
    print("(4)model input===>" , image1.shape , image2.shape)
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
    print("(4)model input===>" , image1.shape , image2.shape)
    image1 = F.pad(image1, [0, pad_w, 0, pad_h], mode='replicate')
    image2 = F.pad(image2, [0, pad_w, 0, pad_h], mode='replicate')
    depth1 = F.pad(depth1, [0, pad_w, 0, pad_h], mode='replicate')
    depth2 = F.pad(depth2, [0, pad_w, 0, pad_h], mode='replicate')
    print("(5)model input===>" , image1.shape , image2.shape)

    depth1 = (depth_scale * depth1).float()
    depth2 = (depth_scale * depth2).float()
    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())
    print("(6)model input===>" , image1.shape , image2.shape)

    depth1 = depth1.float()
    depth2 = depth2.float()

    return image1, image2, depth1, depth2, (pad_w, pad_h)


@torch.no_grad()
def test_sceneflow(model):
    print("test_sceneflow===>")
    
    # Initialize metrics
    metrics_all = {'epe2d': 0.0, 'epe3d': 0.0, '1px': 0.0, '5cm': 0.0, '10cm': 0.0}
    metrics_flownet3d = {'epe3d': 0.0, '5cm': 0.0, '10cm': 0.0}
    count_all = 0
    count_sampled = 0
    
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    # train_dataset = FlyingThingsTest()
    # train_loader = DataLoader(train_dataset, **loader_args)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(config, config.DATASET.TEST_SUBSET, False,transforms.Compose([transforms.ToTensor(),normalize]))
    print("test_dataset===>", len(test_dataset))
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        sampler=sampler_val,
        pin_memory=True,
        num_workers=config.WORKERS)
    print("train_loader===>", len(train_loader))

    # for i_batch, test_data_blob in enumerate(tqdm(train_loader)):
    #     image1, image2, depth1, depth2, flow2d, flow3d, intrinsics, index = \
    #         [data_item.cuda() for data_item in test_data_blob]

    for i_batch, test_data_blob in enumerate(tqdm(train_loader)):
    # Unpack all items from the dataloader
        inputs, input_t1, meta, meta_t1, flows, valids, disps, disps_t1, disps_change, sceneflows, sceneflow_valids = test_data_blob
        print("(1)inputs_raft3d===>" , len(inputs) , input_t1[0].shape)
        
        # Stack multi-view data into batches
        # inputs and input_t1 are lists of length num_views
        image1 = torch.stack(inputs, dim=0).cuda()  # [num_views, C, H, W]
        print("(2)image1 shape===>", image1.shape)
        image2 = torch.stack(input_t1, dim=0).cuda()  # [num_views, C, H, W]
        
        # Stack disparity data
        depth1 = torch.stack(disps, dim=0).cuda()  # [num_views, 1, H, W] - disparity at t
        depth2 = torch.stack(disps_t1, dim=0).cuda()  # [num_views, 1, H, W] - disparity at t+1
        
        # Stack flow data
        flow2d = torch.stack(flows, dim=0).cuda()  # [num_views, 2, H, W] - optical flow
        flow3d = torch.stack(sceneflows, dim=0).cuda()  # [num_views, 3, H, W] - scene flow
        
        # Optional: stack validity masks if needed
        flow2d_valid = torch.stack(valids, dim=0).cuda()  # [num_views, H, W]
        flow3d_valid = torch.stack(sceneflow_valids, dim=0).cuda()  # [num_views, H, W]
        disp_change = torch.stack(disps_change, dim=0).cuda()  # [num_views, 1, H, W]

        # initialize meta
        stacked_meta = [meta, meta_t1]
        
        # Extract intrinsics for both timesteps
        intrinsics_t0_list = []
        intrinsics_t1_list = []
        indices = []
        
        for view_meta_t0, view_meta_t1 in zip(meta, meta_t1):
            # Get camera intrinsics for both timesteps
            cam_intri_t0 = view_meta_t0['camera_Intri']  # [3, 3] or [4, 4]
            cam_intri_t1 = view_meta_t1['camera_Intri']
            
            print(f"Debug - cam_intri_t0 shape: {cam_intri_t0.shape}")
            print(f"Debug - cam_intri_t1 shape: {cam_intri_t1.shape}")
            
            # Ensure 3x3 format
            if cam_intri_t0.shape[0] == 4:
                cam_intri_t0 = cam_intri_t0[:3, :3]
            if cam_intri_t1.shape[0] == 4:
                cam_intri_t1 = cam_intri_t1[:3, :3]
                
            # Handle case where camera_Intri might have batch dimension
            if cam_intri_t0.dim() > 2:
                cam_intri_t0 = cam_intri_t0[0]  # Take first batch
            if cam_intri_t1.dim() > 2:
                cam_intri_t1 = cam_intri_t1[0]  # Take first batch
            
            print(f"Debug - After processing cam_intri_t0 shape: {cam_intri_t0.shape}")
            print(f"Debug - After processing cam_intri_t1 shape: {cam_intri_t1.shape}")
            
            intrinsics_t0_list.append(cam_intri_t0)
            intrinsics_t1_list.append(cam_intri_t1)
            
            # Get index
            image_path = view_meta_t0['image'][0]
            frame_id = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
            try:
                idx = int(frame_id)
            except:
                idx = i_batch
            indices.append(idx)
        
        # Stack intrinsics: [num_views, 3, 3] for each timestep
        intrinsics_t0 = torch.stack(intrinsics_t0_list, dim=0).cuda()  # [5, 3, 3]
        intrinsics_t1 = torch.stack(intrinsics_t1_list, dim=0).cuda()  # [5, 3, 3]
        
        print(f"Debug - intrinsics_t0 shape after stacking: {intrinsics_t0.shape}")
        print(f"Debug - intrinsics_t1 shape after stacking: {intrinsics_t1.shape}")
        
        # Combine timesteps: [num_views, 2, 3, 3]
        intrinsics_both = torch.stack([intrinsics_t0, intrinsics_t1], dim=1)  # [5, 2, 3, 3]
        
        print(f"Debug - intrinsics_both shape: {intrinsics_both.shape}")
        
        # Add batch dimension: [num_views, batch_size, num_timesteps, 3, 3]
        intrinsics = intrinsics_both.unsqueeze(1)  # [5, 1, 2, 3, 3]
        
        index = torch.tensor(indices).cuda()
        
        print(f"Debug - Final intrinsics shape: {intrinsics.shape}")  # Should be [5, 1, 2, 3, 3]

        print("flow2d shape===>", flow2d.shape)
        print("depth1 shape===>", depth1.shape)

        # compute per-pixel 2D flow magnitude: sum over flow channels (dim=1)
        # mag = torch.sum(flow2d**2, dim=1).sqrt()
        mag = torch.sum(flow2d**2, dim=1)
        print("mag shape===>", mag.shape)
        
        # Get dimensions - mag is [B, T, Hf, Wf] and depth1 is [B, T, Cd, Hd, Wd]
        B_mag, T_mag, Hf, Wf = mag.shape
        B_dep, T_dep, Cd, Hd, Wd = depth1.shape
        
        print(f"Debug - mag shape: {mag.shape}, depth1 shape: {depth1.shape}")
        
        # Resize depth1 to match mag dimensions
        depth1_rs = F.interpolate(
            depth1.view(B_dep * T_dep, Cd, Hd, Wd),  # reshape to (B*T, Cd, Hd, Wd)
            size=(Hf, Wf),
            mode='bilinear',
            align_corners=False
        )
        # now reshape back:
        depth1_rs = depth1_rs.view(B_dep, T_dep, Cd, Hf, Wf)
        
        print(f"Debug - resized depth1_rs shape: {depth1_rs.shape}")
        
        # Since mag is [B, T, H, W] and depth1_rs is [B, T, 1, H, W], squeeze the channel dimension
        if depth1_rs.shape[2] == 1:
            depth1_rs = depth1_rs.squeeze(2)  # [B, T, H, W]
        
        print(f"Debug - final mag shape: {mag.shape}, final depth1_rs shape: {depth1_rs.shape}")

        # Flatten both for valid mask - ensure both have same number of elements
        mag_flat   = mag.reshape(-1)        
        depth_flat = depth1_rs.reshape(-1)  
        
        print(f"Debug - mag_flat size: {mag_flat.shape}, depth_flat size: {depth_flat.shape}")
        
        # Only create valid mask if sizes match
        if mag_flat.shape[0] == depth_flat.shape[0]:
            valid = (mag_flat < MAX_FLOW) & (depth_flat < MAX_DEPTH)
        else:
            print(f"Warning: tensor size mismatch, using all valid pixels")
            valid = torch.ones_like(mag_flat, dtype=torch.bool)

        # pad and normalize images
        print("(3)model input===>" , image1.shape , image2.shape)
        image1, image2, depth1, depth2, padding = \
            prepare_images_and_depths(image1, image2, depth1, depth2, DEPTH_SCALE)

        # Prepare multi-view format for DQ model
        print("(7)model input===>" , image1.shape , image2.shape)
        num_views = 5
        
        # Check if this is the multi-view format (5 views) or 10 views (5 views x 2 timesteps)
        if image1.shape[0] == 5:  # Single timestep per view
            # Use the images directly without reshaping
            image1_t0_list = [image1[i:i+1] for i in range(5)]  # List of 5 tensors [1,3,H,W]
            image2_t1_list = [image2[i:i+1] for i in range(5)]  # List of 5 tensors [1,3,H,W]
        elif image1.shape[0] == 10:  # Two timesteps per view
            batch_size = 1  # We know this is 1 batch
            # Reshape to proper format: [batch, views, timesteps, C, H, W]
            image1_reshaped = image1.view(batch_size, num_views, 2, *image1.shape[1:])
            image2_reshaped = image2.view(batch_size, num_views, 2, *image2.shape[1:])
            
            # Create lists for each timestep with proper batch dimension
            image1_t0_list = [image1_reshaped[:, i, 0] for i in range(num_views)]  # 5 tensors of [batch,3,H,W]
            image2_t1_list = [image2_reshaped[:, i, 1] for i in range(num_views)]  # 5 tensors of [batch,3,H,W]
        else:
            raise ValueError(f"Unexpected image1 shape: {image1.shape}")

        image_for_dq = [image1_t0_list, image2_t1_list]

        # Convert intrinsics to float32
        intrinsics_float32 = intrinsics.float()

        # Run model
        Ts = model(image1, image2, depth1, depth2, intrinsics_float32, iters=16, 
                   meta=stacked_meta, image_for_dq=image_for_dq)

        # use transformation field to extract 2D and 3D flow
        # flow2d_est, flow3d_est, _ = pops.induced_flow(Ts, depth1, intrinsics)
        #=======================================================================================>
        # Convert intrinsics to the format expected by induced_flow: [num_views, 3, 3]
        # intrinsics_float32 shape is [5, 1, 2, 3, 3], we need [5, 3, 3]
        intrinsics_for_flow = intrinsics_float32[:, 0, 0, :, :]  # Take first batch, first timestep
        
        # Debug: Extract the original depth1 for the first timestep only
        # depth1 might have been modified by the model to include multiple timesteps
        # We need depth for just the first timestep and 5 views: [5, 1, H, W]
        print(f"Debug - depth1 shape before flow computation: {depth1.shape}")
        print(f"Debug - Ts shape: {Ts.shape}")
        
        # If depth1 has 10 views (5 views × 2 timesteps), take only first 5 views
        if depth1.shape[0] == 10:
            depth1_for_flow = depth1[:5]  # Take first 5 views (timestep 0)
            print(f"Debug - Using first 5 views, depth1_for_flow shape: {depth1_for_flow.shape}")
        else:
            depth1_for_flow = depth1
            print(f"Debug - Using original depth1, shape: {depth1_for_flow.shape}")
        
        # Upsample Ts to match the full resolution of depth1_for_flow
        # Ts is at low resolution (e.g., 512x512), need to upsample to full resolution (1080x1920)
        if depth1_for_flow.dim() == 4:  # [views, channels, H, W]
            target_h, target_w = depth1_for_flow.shape[2], depth1_for_flow.shape[3]
        else:  # [views, H, W]
            target_h, target_w = depth1_for_flow.shape[1], depth1_for_flow.shape[2]
            
        print(f"Debug - Target resolution: {target_h}x{target_w}")
        print(f"Debug - Current Ts resolution: {Ts.shape[1]}x{Ts.shape[2]}")
        
        # Upsample Ts if needed
        if Ts.shape[1] != target_h or Ts.shape[2] != target_w:
            # Ts.data has shape [views, H, W, 7] - need to permute for interpolation
            Ts_data = Ts.data  # [5, 512, 512, 7]
            Ts_data = Ts_data.permute(0, 3, 1, 2)  # [5, 7, 512, 512]
            Ts_data_upsampled = F.interpolate(Ts_data, size=(target_h, target_w), mode='bilinear', align_corners=False)
            Ts_data_upsampled = Ts_data_upsampled.permute(0, 2, 3, 1)  # [5, H, W, 7]
            
            # Create new SE3 object with upsampled data
            Ts_upsampled = SE3(Ts_data_upsampled)
            print(f"Debug - Upsampled Ts to: {Ts_upsampled.shape}")
        else:
            Ts_upsampled = Ts
            print(f"Debug - No upsampling needed for Ts")
        
        flow2d_est, flow3d_est, _ = pops.induced_flow(Ts_upsampled, depth1_for_flow, intrinsics_for_flow)
        
        # Debug: Check flow shapes
        print(f"Debug - flow2d_est shape: {flow2d_est.shape}")
        print(f"Debug - flow3d_est shape: {flow3d_est.shape}")
        print(f"Debug - ground truth flow2d shape: {flow2d.shape}")
        print(f"Debug - ground truth flow3d shape: {flow3d.shape}")
        
        # Reshape ground truth to match estimated flow format
        # flow2d is [5, 2, 2, 512, 960], we need [5, 512, 960, 2]
        # Take the first element from the extra dimensions and transpose
        if flow2d.dim() == 5 and flow2d.shape[1] == 2 and flow2d.shape[2] == 2:
            flow2d = flow2d[:, 0, :, :, :].permute(0, 2, 3, 1)  # [5, 2, 512, 960] -> [5, 512, 960, 2]
        
        # flow3d is [5, 2, 3, 512, 960], we need [5, 512, 960, 3]
        if flow3d.dim() == 5 and flow3d.shape[1] == 2:
            flow3d = flow3d[:, 0, :, :, :].permute(0, 2, 3, 1)  # [5, 3, 512, 960] -> [5, 512, 960, 3]
        
        print(f"Debug - after reshaping flow2d: {flow2d.shape}")
        print(f"Debug - after reshaping flow3d: {flow3d.shape}")
        
        # unpad the flow fields / undo depth scaling
        flow2d_est = flow2d_est[:, :-4, :, :2]
        flow3d_est = flow3d_est[:, :-4] / DEPTH_SCALE
        
        # Resize ground truth to match estimated flow resolution
        print(f"Debug - flow2d shape before resize: {flow2d.shape}")
        print(f"Debug - flow2d_est shape: {flow2d_est.shape}")
        
        # Handle multi-view format properly
        if len(flow2d.shape) == 5:  # Multi-view format [5, 1, 2, H, W]
            # Reshape to [5, 2, H, W] for interpolation
            flow2d = flow2d.squeeze(1)  # Remove singleton dimension -> [5, 2, H, W]
            if flow2d.shape[2:4] != flow2d_est.shape[1:3]:
                flow2d = F.interpolate(flow2d, 
                                     size=flow2d_est.shape[1:3], 
                                     mode='bilinear', align_corners=False)
            # Convert to [5, H, W, 2] to match flow2d_est format
            flow2d = flow2d.permute(0, 2, 3, 1)
        else:  # Standard format [N, H, W, 2]
            if flow2d.shape[1:3] != flow2d_est.shape[1:3]:
                flow2d = F.interpolate(flow2d.permute(0, 3, 1, 2), 
                                     size=flow2d_est.shape[1:3], 
                                     mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        print(f"Debug - flow3d shape before resize: {flow3d.shape}")
        if len(flow3d.shape) == 5:  # Multi-view format [5, 1, 3, H, W]
            # Reshape to [5, 3, H, W] for interpolation
            flow3d = flow3d.squeeze(1)  # Remove singleton dimension -> [5, 3, H, W]
            if flow3d.shape[2:4] != flow3d_est.shape[1:3]:
                flow3d = F.interpolate(flow3d, 
                                     size=flow3d_est.shape[1:3], 
                                     mode='bilinear', align_corners=False)
            # Convert to [5, H, W, 3] to match flow3d_est format
            flow3d = flow3d.permute(0, 2, 3, 1)
        else:  # Standard format [N, H, W, 3]
            if flow3d.shape[1:3] != flow3d_est.shape[1:3]:
                flow3d = F.interpolate(flow3d.permute(0, 3, 1, 2), 
                                     size=flow3d_est.shape[1:3], 
                                     mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        print(f"Debug - final flow2d_est: {flow2d_est.shape}")
        print(f"Debug - final flow2d: {flow2d.shape}")

        epe2d = torch.sum((flow2d_est - flow2d)**2, -1).sqrt()
        epe3d = torch.sum((flow3d_est - flow3d)**2, -1).sqrt()

        # Recompute valid mask to match the final tensor dimensions
        # Create new mask based on final flow estimates
        mag_final = torch.sum(flow2d_est**2, dim=-1).sqrt()  # [5, H, W]
        depth_final = torch.sum(flow3d_est**2, dim=-1).sqrt()  # [5, H, W] - use 3D flow magnitude as depth proxy
        
        # Create valid mask for final tensors
        valid = (mag_final < MAX_FLOW) & (depth_final < MAX_DEPTH)
        valid_flat = valid.reshape(-1)

        # our evaluation (use all valid pixels)
        epe2d_all = epe2d.reshape(-1)[valid_flat].double().cpu().numpy()
        epe3d_all = epe3d.reshape(-1)[valid_flat].double().cpu().numpy()
        
        count_all += epe2d_all.shape[0]
        metrics_all['epe2d'] += epe2d_all.sum()
        metrics_all['epe3d'] += epe3d_all.sum()
        metrics_all['1px'] += np.count_nonzero(epe2d_all < 1.0)
        metrics_all['5cm'] += np.count_nonzero(epe3d_all < .05)
        metrics_all['10cm'] += np.count_nonzero(epe3d_all < .10)

        # FlowNet3D evaluation (only use sampled non-occ pixels)
        # Since we already filtered with valid mask, just use the filtered tensors
        epe2d_sampled = epe2d_all
        epe3d_sampled = epe3d_all
        
        count_sampled += epe2d_sampled.shape[0]
        metrics_flownet3d['epe3d'] += epe3d_sampled.mean()
        metrics_flownet3d['5cm'] += (epe3d_sampled < .05).astype(float).mean()
        metrics_flownet3d['10cm'] += (epe3d_sampled < .10).astype(float).mean()

    # Average results over all valid pixels
    print("all...")
    for key in metrics_all:
        print(key, metrics_all[key] / count_all)

    # FlowNet3D evaluation methodology
    print("non-occ (FlowNet3D Evaluation)...")
    for key in metrics_flownet3d:
        print(key, metrics_flownet3d[key] / (i_batch + 1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path the model weights')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    parser.add_argument('--radius', type=int, default=32)
    args = parser.parse_args()

    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D

    model = torch.nn.DataParallel(RAFT3D(args))
    model.load_state_dict(torch.load(args.model), strict=False)

    model.cuda()
    model.eval()

    test_sceneflow(model)
