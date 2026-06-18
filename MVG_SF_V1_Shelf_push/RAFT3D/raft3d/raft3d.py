import torch
import torch.nn as nn
import torch.nn.functional as F

# lietorch for tangent space backpropogation
from lietorch import SE3

from .blocks.extractor import BasicEncoder
from .blocks.resnet import FPN
from .blocks.corr import CorrBlock
from .blocks.gru import ConvGRU
from .sampler_ops import bilinear_sampler, depth_sampler

from . import projective_ops as pops
from . import se3_field

#================================================================================================
# DQ-RAFT3D Model
#================================================================================================
import sys
import os

# Add root to system path so you can access lib.models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import lib.models.dq_transformer as dq_transformer
from lib.models.dq_transformer import get_mvp
from lib.core.config import config, update_config, update_config_dynamic_input


def init_dq_config(cfg_path='/mnt/MIG_archive24/Datasets/iota/JagritiDatasets/MVG_SF_V1_Shelf/configs/shelf_campus/shelf_knn5-lr4-q1024.yaml', unknown_args=[]):
    update_config(cfg_path)
    update_config_dynamic_input(unknown_args)

#================================================================================================


GRAD_CLIP = .01

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs()>GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.gru = ConvGRU(hidden_dim)

        self.corr_enc = nn.Sequential(
            nn.Conv2d(196, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3*128, 1, padding=0))

        self.flow_enc = nn.Sequential(
            nn.Conv2d(9, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3*128, 1, padding=0))

        self.ae = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, 1, padding=0),
            GradientClip())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip())

        self.weight = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            nn.Sigmoid(),
            GradientClip())

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0),
            GradientClip())


    def forward(self, net, inp, corr, flow, twist, dz, upsample=True):
        motion_info = torch.cat([flow, 10*dz, 10*twist], dim=-1)
        motion_info = motion_info.clamp(-50.0, 50.0).permute(0,3,1,2)

        mot = self.flow_enc(motion_info)
        cor = self.corr_enc(corr)

        net = self.gru(net, inp, cor, mot)

        ae = self.ae(net)
        mask = self.mask(net)
        delta = self.delta(net)
        weight = self.weight(net)

        return net, mask, ae, delta, weight


class RAFT3D(nn.Module):
    def __init__(self, args):
        init_dq_config()
        super(RAFT3D, self).__init__()

        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 3

        # feature network, context network, and update block
        # self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.dq_model = dq_transformer.get_mvp(config, is_train=False) 
        self.cnet = FPN(output_dim=hdim+3*hdim)
        self.update_block = BasicUpdateBlock(args, hidden_dim=hdim)
        
        # Channel adjustment layer to convert DQ features (64 channels) to RAFT format (128 channels)
        self.channel_adjust = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        
        self.dq_model.eval()
        self.dq_model.to('cuda')

    def create_depth_map_from_reference_points_multiview(self, reference_points, intrinsics, image_height, image_width, num_views=5):
        """
        Convert sparse 3D reference points to dense depth maps for multiple camera views
        
        Args:
            reference_points: [B, N*joints, 3] - 3D points in world/camera coordinates
            intrinsics: [num_views, B, 3, 3] or [num_views, B, 4] - camera intrinsics for each view
            image_height, image_width: target depth map dimensions
            num_views: number of camera views (default 5)
        
        Returns:
            depth_maps: [num_views*B, H, W] - dense depth maps for all views
        """
        batch_size = intrinsics.shape[1]  # Get batch size from intrinsics, not reference_points
        device = reference_points.device
        
        print(f"Debug MultiView - reference_points shape: {reference_points.shape}")
        print(f"Debug MultiView - intrinsics shape: {intrinsics.shape}")
        print(f"Debug MultiView - num_views: {num_views}")
        print(f"Debug MultiView - batch_size from intrinsics: {batch_size}")
        
        # Initialize depth maps for all views
        depth_maps = torch.zeros(num_views * batch_size, image_height, image_width, device=device)
        
        # Handle intrinsics shape: [num_views, batch_size, 3, 3] or [num_views, batch_size, 4]
        if intrinsics.dim() == 4:  # [num_views, batch_size, 3, 3]
            intrinsics_reshaped = intrinsics.view(num_views * batch_size, 3, 3)
        elif intrinsics.dim() == 3:  # [num_views, batch_size, 4] 
            intrinsics_reshaped = intrinsics.view(num_views * batch_size, -1)
        else:
            raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
        
        # Process each view-batch combination
        view_batch_idx = 0
        for view in range(num_views):
            for batch in range(batch_size):
                # For reference points, we need to handle the case where reference_points
                # might have different batch dimension than intrinsics batch dimension
                if reference_points.shape[0] > batch:
                    points_3d = reference_points[batch]  # [N*joints, 3]
                else:
                    # Use the first (or only) set of reference points for all batches
                    points_3d = reference_points[0]  # [N*joints, 3]
                
                # Get intrinsics for this specific view-batch combination
                if intrinsics.dim() == 4:  # [num_views, batch_size, 3, 3]
                    intrinsic_matrix = intrinsics[view, batch]  # [3, 3]
                    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
                    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
                else:  # [num_views, batch_size, 4+]
                    intrinsic = intrinsics[view, batch]  # [4] or more
                    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
                
                # Project 3D points to 2D image coordinates for this camera view
                X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
                
                # Skip points with zero or negative depth
                valid_depth_mask = Z > 0.1  # minimum depth threshold
                if not valid_depth_mask.any():
                    view_batch_idx += 1
                    continue
                    
                X, Y, Z = X[valid_depth_mask], Y[valid_depth_mask], Z[valid_depth_mask]
                
                # Project to pixel coordinates for this camera view
                u = (fx * X / Z + cx).round().long()
                v = (fy * Y / Z + cy).round().long()
                
                # Filter valid projections
                valid_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
                
                if valid_mask.any():
                    valid_u = u[valid_mask]
                    valid_v = v[valid_mask]
                    valid_z = Z[valid_mask]
                    
                    # Fill depth map at projected locations
                    for i in range(len(valid_u)):
                        curr_u, curr_v, curr_z = valid_u[i], valid_v[i], valid_z[i]
                        # Take minimum depth if multiple points project to same pixel
                        if depth_maps[view_batch_idx, curr_v, curr_u] == 0:
                            depth_maps[view_batch_idx, curr_v, curr_u] = curr_z
                        else:
                            depth_maps[view_batch_idx, curr_v, curr_u] = min(depth_maps[view_batch_idx, curr_v, curr_u], curr_z)
                
                view_batch_idx += 1
        
        # Fill gaps using interpolation
        depth_maps = self.interpolate_depth_gaps(depth_maps)
        
        return depth_maps.float()



    def interpolate_depth_gaps(self, depth_map):
        """Fill gaps in sparse depth map using interpolation"""
        # Simple approach: use nearest neighbor for empty pixels
        
        for b in range(depth_map.shape[0]):
            depth_slice = depth_map[b]
            mask = depth_slice > 0
            
            if mask.sum() > 4:  # Need at least 4 points for interpolation
                # Get coordinates of valid depth values
                valid_coords = torch.nonzero(mask, as_tuple=False).float().to(depth_map.device)
                valid_depths = depth_slice[mask]
                
                # Create grid for empty pixels only
                empty_mask = depth_slice == 0
                if empty_mask.sum() > 0:
                    empty_coords = torch.nonzero(empty_mask, as_tuple=False).float().to(depth_map.device)
                    
                    # Simple nearest neighbor interpolation for empty pixels
                    if len(valid_coords) > 0 and len(empty_coords) > 0:
                        # Compute distances between empty pixels and valid pixels
                        distances = torch.cdist(empty_coords, valid_coords)
                        nearest_indices = distances.argmin(dim=1)
                        interpolated_depths = valid_depths[nearest_indices]
                        
                        # Fill empty pixels
                        empty_v = empty_coords[:, 0].long()
                        empty_u = empty_coords[:, 1].long()
                        depth_map[b, empty_v, empty_u] = interpolated_depths
            elif mask.sum() > 0:
                # If very few valid points, just fill with mean depth
                mean_depth = depth_slice[mask].mean()
                depth_map[b] = torch.where(depth_map[b] == 0, mean_depth, depth_map[b])
        
        return depth_map.float()

    # def initializer(self, image):
    #     """ Initialize coords and transformation maps """
    #     # image1 = image[0][0]

    #     batch_size, ch, ht, wd = image1.shape
    #     device = image1.device

    #     y0, x0 = torch.meshgrid(torch.arange(ht//8), torch.arange(wd//8))
    #     coords0 = torch.stack([x0, y0], dim=-1).float()
    #     coords0 = coords0[None].repeat(batch_size, 1, 1, 1).to(device)

    #     Ts = SE3.Identity(batch_size, ht//8, wd//8, device=device)
    #     return Ts, coords0
    def initializer(self, image1, aggregated=True, batch_size_override=None):
        """ Initialize coords and transformation maps 
        
        Args:
            image1: Raw multi-view images [num_views, C, H, W]
            aggregated: If True, we're using aggregated features
            batch_size_override: Override batch size (use actual batch from aggregated features)
        """

        batch_size, ch, ht, wd = image1.shape
        device = image1.device

        if aggregated and batch_size_override is not None:
            # Use the actual batch size from aggregated features
            # This handles cases where batch_size > 1 (e.g., multiple samples or timesteps)
            effective_batch_size = batch_size_override
            print(f"✓ Aggregated mode: Using batch_size={effective_batch_size} from aggregated features")
        elif aggregated:
            # Fallback: assume batch_size=1
            effective_batch_size = 1
            print(f"✓ Aggregated mode: Using batch_size=1 for unified scene flow")
        else:
            # Multi-view mode: separate scene flow per view
            effective_batch_size = batch_size
            print(f"✓ Multi-view mode: Using batch_size={batch_size} for per-view flows")

        # SE3 field initialization needs to match depth map dimensions (64x64)
        # Since we changed depth sampling to 64x64, we need SE3 to match  
        depth_ht, depth_wd = 64, 64  # Match our new depth map dimensions
        
        y0, x0 = torch.meshgrid(torch.arange(depth_ht), torch.arange(depth_wd))
        coords0 = torch.stack([x0, y0], dim=-1).float()
        coords0 = coords0[None].repeat(effective_batch_size, 1, 1, 1).to(device)

        Ts = SE3.Identity(effective_batch_size, depth_ht, depth_wd, device=device)
        
        print(f"DEBUG initializer: image1.shape = {image1.shape}")
        print(f"DEBUG initializer: effective_batch_size = {effective_batch_size}")
        print(f"DEBUG initializer: Ts.data.shape = {Ts.data.shape}")
        print(f"DEBUG initializer: coords0.shape = {coords0.shape}")
        
        return Ts, coords0
        
    # def features_and_correlation(self, image , meta):
    # def features_and_correlation(self, image1, image2 , meta=None):
    def features_and_correlation(self, image1, image2, meta=None, image_for_dq=None):
        """
        Args:
        image1, image2: flattened images [10, 3, H, W] for RAFT3D processing
        meta: metadata
        image_for_dq: [image1_t0_list, image2_t1_list] where each list contains 5 tensors
                      of shape [1, 3, H, W] or [3, H, W]
        """
        print("(10)model input===>" , image1.shape , image2.shape)
        
        # Get DQ features - AGGREGATED CROSS-ATTENTION FEATURES
        dq_output = self.dq_model(image_for_dq, meta)
        out = dq_output if not isinstance(dq_output, tuple) else dq_output[0]
        
        # DQ now returns AGGREGATED features after cross-attention across all 5 views
        # fmap1_dq: [batch, 64, 64, 64] for timestep 0 (aggregated from 5 views)
        # fmap2_dq: [batch, 64, 64, 64] for timestep 1 (aggregated from 5 views)
        fmap1_dq = out['attn_feature_views_0']  
        fmap2_dq = out['attn_feature_views_1']
        
        print(f"✓ Per-view DQ feature shapes: fmap1_dq={fmap1_dq.shape}, fmap2_dq={fmap2_dq.shape}")
        
        # Use aggregated features directly
        # Each feature now contains information from all 5 views via cross-attention + aggregation
        fmap1 = fmap1_dq  # [batch, 64, 64, 64]
        fmap2 = fmap2_dq  # [batch, 64, 64, 64]
        
        # Adjust channels from 64 to 128 to match RAFT3D expectations
        fmap1 = self.channel_adjust(fmap1)  # [batch, 64, 64, 64] -> [batch, 128, 64, 64]
        fmap2 = self.channel_adjust(fmap2)  # [batch, 64, 64, 64] -> [batch, 128, 64, 64]
        
        print(f"✓ Channel-adjusted DQ feature shapes: fmap1={fmap1.shape}, fmap2={fmap2.shape}")
        
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # extract context features using Resnet50
        # Use first image from image1 (they should all be similar after preprocessing)
        # We only need one context since we have aggregated multi-view features
        if image1.shape[0] > 1:
            # Take first image as context
            context_image = image1[0:1]  # [1, 3, H, W]
        else:
            context_image = image1
            
        print(f"DEBUG: context_image.shape = {context_image.shape}")
        net_inp = self.cnet(context_image)
        print(f"DEBUG: net_inp.shape = {net_inp.shape}")
        net, inp = net_inp.split([128, 128*3], dim=1)
        print(f"DEBUG: net.shape = {net.shape}, inp.shape = {inp.shape}")

        net = torch.tanh(net)
        inp = torch.relu(inp)
        
        # Downsample inp to match the 64x64 feature map dimensions
        if inp.shape[-1] != 64:
            inp = F.interpolate(inp, size=(64, 64), mode='bilinear', align_corners=True)
            print(f"DEBUG: downsampled inp.shape = {inp.shape}")
        
        # Also downsample net to match
        if net.shape[-1] != 64:
            net = F.interpolate(net, size=(64, 64), mode='bilinear', align_corners=True)
            print(f"DEBUG: downsampled net.shape = {net.shape}")

        return corr_fn, net, inp

    def forward(self, image1, image2, depth1, depth2, intrinsics, iters=12, train_mode=False, meta=None, image_for_dq=None):
    # def forward(self, image , meta ,depth1, depth2, intrinsics, iters=12, train_mode=False):
    # def forward(self, image , meta, intrinsics, iters=12, train_mode=False):
        """ Estimate optical flow between pair of frames """

        # image1 = image[0][0]
        # image2 = image[1][0]
        print("(8)model input===>" , image1.shape , image2.shape)
        
        # First, get DQ output to determine actual batch size
        dq_output_temp = self.dq_model(image_for_dq, meta)
        out_temp = dq_output_temp if not isinstance(dq_output_temp, tuple) else dq_output_temp[0]
        actual_batch_size = out_temp['attn_feature_views_0'].shape[0]
        
        print(f"✓ Detected batch size from aggregated features: {actual_batch_size}")
        
        # Use aggregated mode with actual batch size
        Ts, coords0 = self.initializer(image1, aggregated=True, batch_size_override=actual_batch_size)
        
        print("(9)model input===>" , image1.shape , image2.shape)
        # print("meta0===>", len(meta0) , meta0)
        # print("meta00===>" , meta0[0]['center'].shape)
        # print("meta10===>" , meta1[0]['center'].shape)
        corr_fn, net, inp = self.features_and_correlation(image1, image2, meta=meta, image_for_dq=image_for_dq)

        # intrinsics and depth at 1/8 resolution - Multi-view handling
        original_intrinsics = intrinsics  # Keep original for multi-view depth creation
        
        print(f"Debug - Input intrinsics shape: {intrinsics.shape}")
        print(f"Debug - Input image1 shape: {image1.shape}")
        
        # Get DQ transformer output first
        dq_output = self.dq_model(image_for_dq, meta) 
        out = dq_output if not isinstance(dq_output, tuple) else dq_output[0]

        # Extract reference points and aggregated features
        ref_points_0 = out['reference_points0'].float()  # [B, N*joints, 3] - ensure float32
        ref_points_1 = out['reference_points1'].float()  # [B, N*joints, 3] - ensure float32
        aggregated_feat_0 = out['attn_feature_views_0']  # [B, 64, 64, 64]
        
        # Get actual batch size from aggregated features
        actual_batch_size = aggregated_feat_0.shape[0]
        
        print(f"Debug - Reference points shape: {ref_points_0.shape}")
        print(f"Debug - Aggregated feature batch size: {actual_batch_size}")
        
        # Handle multi-view depth map creation
        _, _, H, W = image1.shape
        
        # AGGREGATED MODE: Create depth maps for actual batch size
        # Use first/reference camera's intrinsics for unified depth
        if intrinsics.dim() == 5:  # [num_views, batch, timesteps, 3, 3]
            num_views, batch_size_intrinsics, num_timesteps = intrinsics.shape[:3]
            print(f"Debug - Multi-view with timesteps: {num_views} views, {batch_size_intrinsics} batches, {num_timesteps} timesteps")
            
            # For aggregated scene flow: use first camera as reference
            # Shape: [5, 1, 2, 3, 3] -> [1, batch, 3, 3] by taking first view, t=0
            intrinsics_ref = intrinsics[0:1, :, 0, :, :]  # [1, batch, 3, 3] - first camera
            
            print(f"✓ Aggregated mode: Using first camera as reference")
            print(f"  - Reference intrinsics shape: {intrinsics_ref.shape}")
            print(f"  - Actual batch size for depth: {actual_batch_size}")
            
            # Create depth maps for actual batch size using reference camera
            # num_views=1 means single camera, but batch_size can be > 1
            depth1 = self.create_depth_map_from_reference_points_multiview(
                ref_points_0, intrinsics_ref, H, W, num_views=1)
            depth2 = self.create_depth_map_from_reference_points_multiview(
                ref_points_1, intrinsics_ref, H, W, num_views=1)
            
            # For RAFT3D operations: [actual_batch_size, 3, 3]
            intrinsics_3x3 = intrinsics_ref.view(-1, 3, 3)  # [batch, 3, 3]
        elif intrinsics.dim() == 4:  # Multi-view case: [num_views, batch, 3, 3]
            num_views, batch_size_intrinsics = intrinsics.shape[0], intrinsics.shape[1]
            print(f"Debug - Multi-view: {num_views} views, {batch_size_intrinsics} batches")
            
            # Aggregated mode: use first camera as reference
            intrinsics_ref = intrinsics[0:1, :, :, :]  # [1, batch, 3, 3]
            
            print(f"✓ Aggregated mode: Using first camera as reference")
            print(f"  - Actual batch size for depth: {actual_batch_size}")
            
            # Create depth maps for actual batch size
            depth1 = self.create_depth_map_from_reference_points_multiview(
                ref_points_0, intrinsics_ref, H, W, num_views=1)
            depth2 = self.create_depth_map_from_reference_points_multiview(
                ref_points_1, intrinsics_ref, H, W, num_views=1)
            
            # For RAFT3D: [batch, 3, 3]
            intrinsics_3x3 = intrinsics_ref.view(-1, 3, 3)
            
        elif intrinsics.dim() == 3:  # [batch, 3, 3] or [num_views, 3, 3]
            # Aggregated mode: use first camera
            intrinsics_ref = intrinsics[0:1, :, :]  # [1, 3, 3]
            intrinsics_3x3 = intrinsics_ref
            
            print(f"✓ Aggregated mode: Using first camera from shape {intrinsics.shape}")
            print(f"  - Actual batch size for depth: {actual_batch_size}")
            
            # Create depth maps with single reference
            depth1 = self.create_depth_map_from_reference_points_multiview(
                ref_points_0, intrinsics_ref.unsqueeze(0), H, W, num_views=1)
            depth2 = self.create_depth_map_from_reference_points_multiview(
                ref_points_1, intrinsics_ref.unsqueeze(0), H, W, num_views=1)
        else:
            raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")


        # Extract intrinsic parameters for RAFT3D
        # After aggregation: intrinsics_3x3 is [batch, 3, 3]
        fx = intrinsics_3x3[:, 0, 0]  # [batch]
        fy = intrinsics_3x3[:, 1, 1]  # [batch]
        cx = intrinsics_3x3[:, 0, 2]  # [batch]
        cy = intrinsics_3x3[:, 1, 2]  # [batch]
        intrinsics_for_raft = torch.stack([fx, fy, cx, cy], dim=1)  # [batch, 4]
        
        print(f"✓ Created depth maps: depth1 {depth1.shape}, depth2 {depth2.shape}")
        print(f"✓ RAFT intrinsics: {intrinsics_for_raft.shape}")
        print(f"✓ Batch size consistency: Ts={actual_batch_size}, depth={depth1.shape[0]}, intrinsics={intrinsics_for_raft.shape[0]}")
        
        # CRITICAL FIX: Ensure Ts batch size matches depth map batch size
        # This prevents broadcasting errors in lietorch when Ts * X0 is computed
        depth_batch_size = depth1.shape[0]
        if Ts.shape[0] != depth_batch_size:
            print(f"⚠️  Batch size mismatch detected! Reinitializing Ts:")
            print(f"   - Ts batch size: {Ts.shape[0]}")
            print(f"   - Depth batch size: {depth_batch_size}")
            
            # Reinitialize Ts with correct batch size
            device = Ts.device
            depth_ht, depth_wd = 64, 64
            Ts = SE3.Identity(depth_batch_size, depth_ht, depth_wd, device=device)
            
            # Also update coords0 to match
            y0, x0 = torch.meshgrid(
                torch.arange(depth_ht, device=device), 
                torch.arange(depth_wd, device=device))
            coords0 = torch.stack([x0, y0], dim=-1).float()
            coords0 = coords0[None].repeat(depth_batch_size, 1, 1, 1)
            
            print(f"✓ Reinitialized Ts with batch_size={depth_batch_size}")
        
        print(f"✓ Final batch size consistency: Ts={Ts.shape[0]}, depth={depth1.shape[0]}, intrinsics={intrinsics_for_raft.shape[0]}")
        print(f"✓ All tensors ready for aggregated scene flow computation")
        
        # Calculate sampling to get exactly 64x64 from input size
        H, W = depth1.shape[1], depth1.shape[2]  # Actual depth map dimensions
        h_step = H // 64  # Calculate step size
        w_step = W // 64  # Calculate step size
        
        # Scale intrinsics according to actual downsampling ratios
        # fx, fy need to be scaled by the downsampling factors
        intrinsics_r8 = intrinsics_for_raft.clone()
        intrinsics_r8[:, 0] = intrinsics_r8[:, 0] / w_step  # fx scaling  
        intrinsics_r8[:, 1] = intrinsics_r8[:, 1] / h_step  # fy scaling
        intrinsics_r8[:, 2] = intrinsics_r8[:, 2] / w_step  # cx scaling
        intrinsics_r8[:, 3] = intrinsics_r8[:, 3] / h_step  # cy scaling
        
        # Downsample depth maps to match feature dimensions and ensure float32
        depth1_r8 = depth1[:, ::h_step, ::w_step].float()
        depth2_r8 = depth2[:, ::h_step, ::w_step].float()
        
        # CRITICAL FIX: Reinitialize Ts to match actual downsampled depth spatial dimensions
        # This prevents broadcasting errors when Ts * X0 is computed
        actual_depth_ht, actual_depth_wd = depth1_r8.shape[1], depth1_r8.shape[2]
        if Ts.shape[1] != actual_depth_ht or Ts.shape[2] != actual_depth_wd:
            print(f"⚠️  Spatial dimension mismatch detected! Reinitializing Ts:")
            print(f"   - Ts spatial dims: {Ts.shape[1]}x{Ts.shape[2]}")
            print(f"   - Depth spatial dims: {actual_depth_ht}x{actual_depth_wd}")
            
            # Reinitialize Ts with correct spatial dimensions
            device = Ts.device
            depth_batch_size = depth1_r8.shape[0]
            Ts = SE3.Identity(depth_batch_size, actual_depth_ht, actual_depth_wd, device=device)
            
            # Also update coords0 to match
            y0, x0 = torch.meshgrid(
                torch.arange(actual_depth_ht, device=device), 
                torch.arange(actual_depth_wd, device=device))
            coords0 = torch.stack([x0, y0], dim=-1).float()
            coords0 = coords0[None].repeat(depth_batch_size, 1, 1, 1)
            
            print(f"✓ Reinitialized Ts with spatial dims {actual_depth_ht}x{actual_depth_wd}, batch_size={depth_batch_size}")
        
        print(f"✓ Final dimensions: Ts={Ts.shape}, depth1_r8={depth1_r8.shape}, depth2_r8={depth2_r8.shape}")
        

        flow_est_list = []
        flow_rev_list = []

        for itr in range(iters):
            Ts = Ts.detach()

            # Debug shape information before projective transform
            print(f"DEBUG iteration {itr}: depth1_r8.shape = {depth1_r8.shape}")
            print(f"DEBUG iteration {itr}: intrinsics_r8.shape = {intrinsics_r8.shape}")
            print(f"DEBUG iteration {itr}: Ts data shape = {Ts.data.shape}")
            
            coords1_xyz, _ = pops.projective_transform(Ts, depth1_r8, intrinsics_r8)
            
            coords1, zinv_proj = coords1_xyz.split([2,1], dim=-1)
            zinv, _ = depth_sampler(1.0/depth2_r8, coords1)

            # Debug: check coordinate and depth dimensions
            print(f"Debug coords1 shape: {coords1.shape}")
            print(f"Debug depth1_r8 shape: {depth1_r8.shape}")
            coords1_for_corr = coords1.permute(0,3,1,2).contiguous()
            print(f"Debug coords1_for_corr shape: {coords1_for_corr.shape}")
            
            # CRITICAL FIX: Resize coordinates to match feature dimensions (64x64)
            # Features are always 64x64, but depth downsampling may give different sizes (e.g., 67x67)
            # Correlation function expects coordinates to match feature spatial dimensions
            feature_ht, feature_wd = 64, 64
            if coords1_for_corr.shape[2] != feature_ht or coords1_for_corr.shape[3] != feature_wd:
                print(f"⚠️  Resizing coordinates from {coords1_for_corr.shape[2]}x{coords1_for_corr.shape[3]} to {feature_ht}x{feature_wd}")
                # Interpolate coordinates to match feature dimensions
                # coords1_for_corr is [B, 2, H, W], we need to resize to [B, 2, 64, 64]
                coords1_for_corr = F.interpolate(
                    coords1_for_corr, 
                    size=(feature_ht, feature_wd), 
                    mode='bilinear', 
                    align_corners=False
                )
                print(f"✓ Resized coords1_for_corr to {coords1_for_corr.shape}")
            
            corr = corr_fn(coords1_for_corr)
            print(f"DEBUG iteration {itr}: corr shape after corr_fn = {corr.shape}")
            print(f"DEBUG iteration {itr}: corr expected channels = 196, got {corr.shape[1]}")
            
            flow = coords1 - coords0

            dz = zinv.unsqueeze(-1) - zinv_proj
            twist = Ts.log()

            # CRITICAL FIX: Resize flow and dz to match feature dimensions (64x64)
            # Update block expects all inputs to have same spatial dimensions as features
            feature_ht, feature_wd = 64, 64
            if flow.shape[1] != feature_ht or flow.shape[2] != feature_wd:
                print(f"⚠️  Resizing flow from {flow.shape[1]}x{flow.shape[2]} to {feature_ht}x{feature_wd}")
                # flow is [B, H, W, 2], need to permute to [B, 2, H, W] for interpolation
                flow = flow.permute(0, 3, 1, 2).contiguous()
                flow = F.interpolate(flow, size=(feature_ht, feature_wd), mode='bilinear', align_corners=False)
                flow = flow.permute(0, 2, 3, 1).contiguous()  # Back to [B, H, W, 2]
                print(f"✓ Resized flow to {flow.shape}")
            
            if dz.shape[1] != feature_ht or dz.shape[2] != feature_wd:
                print(f"⚠️  Resizing dz from {dz.shape[1]}x{dz.shape[2]} to {feature_ht}x{feature_wd}")
                # dz is [B, H, W, 1], need to permute to [B, 1, H, W] for interpolation
                dz = dz.permute(0, 3, 1, 2).contiguous()
                dz = F.interpolate(dz, size=(feature_ht, feature_wd), mode='bilinear', align_corners=False)
                dz = dz.permute(0, 2, 3, 1).contiguous()  # Back to [B, H, W, 1]
                print(f"✓ Resized dz to {dz.shape}")
            
            # Also resize twist if needed (twist comes from Ts.log() which has same spatial dims as Ts)
            if twist.shape[1] != feature_ht or twist.shape[2] != feature_wd:
                print(f"⚠️  Resizing twist from {twist.shape[1]}x{twist.shape[2]} to {feature_ht}x{feature_wd}")
                # twist is [B, H, W, 6], need to permute to [B, 6, H, W] for interpolation
                twist = twist.permute(0, 3, 1, 2).contiguous()
                twist = F.interpolate(twist, size=(feature_ht, feature_wd), mode='bilinear', align_corners=False)
                twist = twist.permute(0, 2, 3, 1).contiguous()  # Back to [B, H, W, 6]
                print(f"✓ Resized twist to {twist.shape}")

            net, mask, ae, delta, weight = \
                self.update_block(net, inp, corr, flow, dz, twist)

            # CRITICAL FIX: Resize mask to match Ts dimensions for upsampling operations
            # mask comes from update_block (64x64), but upsampling expects it to match Ts spatial dims
            Ts_ht, Ts_wd = Ts.shape[1], Ts.shape[2]
            if mask.shape[2] != Ts_ht or mask.shape[3] != Ts_wd:
                print(f"⚠️  Resizing mask from {mask.shape[2]}x{mask.shape[3]} to {Ts_ht}x{Ts_wd}")
                # mask is [B, C, H, W], resize to match Ts spatial dims
                mask = F.interpolate(mask, size=(Ts_ht, Ts_wd), mode='bilinear', align_corners=False)
                print(f"✓ Resized mask to {mask.shape}")

            # CRITICAL FIX: Resize delta to match coords1_xyz dimensions
            # coords1_xyz has spatial dims from depth1_r8 (e.g., 67x67)
            # delta has spatial dims from update_block (64x64 after resizing inputs)
            coords1_xyz_ht, coords1_xyz_wd = coords1_xyz.shape[1], coords1_xyz.shape[2]
            if delta.shape[2] != coords1_xyz_ht or delta.shape[3] != coords1_xyz_wd:
                print(f"⚠️  Resizing delta from {delta.shape[2]}x{delta.shape[3]} to {coords1_xyz_ht}x{coords1_xyz_wd}")
                # delta is [B, C, H, W], resize to match coords1_xyz spatial dims
                delta = F.interpolate(delta, size=(coords1_xyz_ht, coords1_xyz_wd), mode='bilinear', align_corners=False)
                print(f"✓ Resized delta to {delta.shape}")

            target = coords1_xyz.permute(0,3,1,2) + delta
            target = target.contiguous()

            # CRITICAL FIX: Resize ae and weight to match target/depth dimensions
            # These are used in se3_field.step_inplace which expects matching spatial dims
            if ae.shape[2] != coords1_xyz_ht or ae.shape[3] != coords1_xyz_wd:
                print(f"⚠️  Resizing ae from {ae.shape[2]}x{ae.shape[3]} to {coords1_xyz_ht}x{coords1_xyz_wd}")
                ae = F.interpolate(ae, size=(coords1_xyz_ht, coords1_xyz_wd), mode='bilinear', align_corners=False)
                print(f"✓ Resized ae to {ae.shape}")
            
            if weight.shape[2] != coords1_xyz_ht or weight.shape[3] != coords1_xyz_wd:
                print(f"⚠️  Resizing weight from {weight.shape[2]}x{weight.shape[3]} to {coords1_xyz_ht}x{coords1_xyz_wd}")
                weight = F.interpolate(weight, size=(coords1_xyz_ht, coords1_xyz_wd), mode='bilinear', align_corners=False)
                print(f"✓ Resized weight to {weight.shape}")

            # Gauss-Newton step
            # Ts = se3_field.step(Ts, ae, target, weight, depth1_r8, intrinsics_r8)
            Ts = se3_field.step_inplace(Ts, ae, target, weight, depth1_r8, intrinsics_r8)

            if train_mode:
                flow2d_rev = target.permute(0,2,3,1)[...,:2] - coords0
                flow2d_rev = se3_field.cvx_upsample(8 * flow2d_rev, mask)

                Ts_up = se3_field.upsample_se3(Ts, mask)
                flow2d_est, flow3d_est, valid = pops.induced_flow(Ts_up, depth1, intrinsics)

                flow_est_list.append(flow2d_est)
                flow_rev_list.append(flow2d_rev)

        if train_mode:
            return flow_est_list, flow_rev_list

        # mask is already resized in the loop, so we can use it directly here
        Ts_up = se3_field.upsample_se3(Ts, mask)
        return Ts_up

