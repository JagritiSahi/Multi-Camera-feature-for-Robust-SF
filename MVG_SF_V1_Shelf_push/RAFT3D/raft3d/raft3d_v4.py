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


def init_dq_config(cfg_path='configs/panoptic/generalization/CMU0ex3.yaml', unknown_args=[]):
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
    def initializer(self, image1):
        """ Initialize coords and transformation maps """

        batch_size, ch, ht, wd = image1.shape
        device = image1.device

        # SE3 field initialization needs to match depth map dimensions (64x64)
        # Since we changed depth sampling to 64x64, we need SE3 to match  
        depth_ht, depth_wd = 64, 64  # Match our new depth map dimensions
        
        y0, x0 = torch.meshgrid(torch.arange(depth_ht), torch.arange(depth_wd))
        coords0 = torch.stack([x0, y0], dim=-1).float()
        coords0 = coords0[None].repeat(batch_size, 1, 1, 1).to(device)

        Ts = SE3.Identity(batch_size, depth_ht, depth_wd, device=device)
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
        
        # Get DQ features - these are consolidated across all views
        dq_output = self.dq_model(image_for_dq, meta)
        out = dq_output if not isinstance(dq_output, tuple) else dq_output[0]
        
        # DQ features are [batch, 64, 64] but we need [num_views*batch, 64, 64]
        fmap1_dq = out['attn_feature_views_0']  # [2, 64, 64] for timestep 0
        fmap2_dq = out['attn_feature_views_1']  # [2, 64, 64] for timestep 1
        
        print(f"Original DQ feature shapes: fmap1_dq={fmap1_dq.shape}, fmap2_dq={fmap2_dq.shape}")
        
        # For multi-view: replicate DQ features for each view
        # image1 shape is [10, 3, H, W] = [5 views × 2 batches, 3, H, W]
        num_total_samples = image1.shape[0]  # 10
        num_views = 5
        batch_size = num_total_samples // num_views  # Should be 2
        
        # Reshape and replicate: [2, 64, 64] -> [10, 64, 64]
        # For each batch, repeat the feature map 5 times (one per view)
        fmap1_list = []
        fmap2_list = []
        
        for b in range(batch_size):  # Loop over 2 batches
            # Replicate this batch's features for all 5 views
            fmap1_batch = fmap1_dq[b].unsqueeze(0)  # [1, 64, 64]
            fmap2_batch = fmap2_dq[b].unsqueeze(0)  # [1, 64, 64]
            
            fmap1_replicated = fmap1_batch.repeat(num_views, 1, 1, 1)  # [5, 64, 64]
            fmap2_replicated = fmap2_batch.repeat(num_views, 1, 1, 1)  # [5, 64, 64]
            
            fmap1_list.append(fmap1_replicated)
            fmap2_list.append(fmap2_replicated)
        
        # Concatenate all batches: [5, 64, 64] + [5, 64, 64] -> [10, 64, 64]
        fmap1 = torch.cat(fmap1_list, dim=0)  # [10, 64, 64, 64]
        fmap2 = torch.cat(fmap2_list, dim=0)  # [10, 64, 64, 64]
        
        # Adjust channels from 64 to 128 to match RAFT3D expectations
        fmap1 = self.channel_adjust(fmap1)  # [10, 64, 64, 64] -> [10, 128, 64, 64]
        fmap2 = self.channel_adjust(fmap2)  # [10, 64, 64, 64] -> [10, 128, 64, 64]
        
        print(f"Channel-adjusted DQ feature shapes: fmap1={fmap1.shape}, fmap2={fmap2.shape}")
        
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # extract context features using Resnet50
        print(f"DEBUG: image1.shape = {image1.shape}")
        net_inp = self.cnet(image1)
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
        Ts, coords0 = self.initializer(image1)
        
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

        # Extract reference points
        ref_points_0 = out['reference_points0'].float()  # [B, N*joints, 3] - ensure float32
        ref_points_1 = out['reference_points1'].float()  # [B, N*joints, 3] - ensure float32
        
        print(f"Debug - Reference points shape: {ref_points_0.shape}")
        
        # Handle multi-view depth map creation
        _, _, H, W = image1.shape
            # Handle different intrinsics shapes
        if intrinsics.dim() == 5:  # [num_views, batch, timesteps, 3, 3]
            num_views, batch_size, num_timesteps = intrinsics.shape[:3]
            print(f"Debug - Multi-view with timesteps: {num_views} views, {batch_size} batches, {num_timesteps} timesteps")
            
            # For depth map creation, we can use timestep 0 intrinsics
            # Shape: [5, 1, 2, 3, 3] -> [5, 1, 3, 3] by taking t=0
            intrinsics_t0 = intrinsics[:, :, 0, :, :]  # [5, 1, 3, 3]
            
            # Create depth maps using t=0 intrinsics
            depth1 = self.create_depth_map_from_reference_points_multiview(
                ref_points_0, intrinsics_t0, H, W, num_views)
            depth2 = self.create_depth_map_from_reference_points_multiview(
                ref_points_1, intrinsics_t0, H, W, num_views)
            
            # For RAFT3D operations, flatten to [num_views*batch, 3, 3]
            intrinsics_3x3 = intrinsics_t0.view(-1, 3, 3)
        elif intrinsics.dim() == 4:  # Multi-view case: [num_views, batch, 3, 3]
            num_views, batch_size = intrinsics.shape[0], intrinsics.shape[1]
            print(f"Debug - Multi-view: {num_views} views, {batch_size} batches")
            
            # Create depth maps for all view-batch combinations
            depth1 = self.create_depth_map_from_reference_points_multiview(
                ref_points_0, intrinsics, H, W, num_views)
            depth2 = self.create_depth_map_from_reference_points_multiview(
                ref_points_1, intrinsics, H, W, num_views)
            
            # Create intrinsics for RAFT3D: [num_views*batch, 4]
            #=====================================================================================================
            # intrinsics_3x3 = intrinsics[:, 0, :, :]  # [num_views, 3, 3] - take first timestep    #====>changed
            intrinsics_3x3 = intrinsics.view(-1, 3, 3)
        elif intrinsics.dim() == 3:  # [batch, 3, 3]
            batch_size = intrinsics.shape[0]
            num_views = image1.shape[0] // batch_size
            
            # Repeat intrinsics for each view
            intrinsics_3x3 = intrinsics.unsqueeze(0).repeat(num_views, 1, 1, 1).view(-1, 3, 3)
            
            # Create depth maps
            depth1 = self.create_depth_map_from_reference_points_multiview(
                ref_points_0, intrinsics.unsqueeze(0), H, W, 1)
            depth2 = self.create_depth_map_from_reference_points_multiview(
                ref_points_1, intrinsics.unsqueeze(0), H, W, 1)
        else:
            raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")


        fx = intrinsics_3x3[:, 0, 0]  # [num_views]
        fy = intrinsics_3x3[:, 1, 1]  # [num_views]
        cx = intrinsics_3x3[:, 0, 2]  # [num_views]
        cy = intrinsics_3x3[:, 1, 2]  # [num_views]
            # intrinsics_4d = torch.stack([fx, fy, cx, cy], dim=1)  # [num_views, 4]
        # intrinsics_for_raft = intrinsics_4d.repeat(batch_size, 1)  # [num_views*batch, 4]
        intrinsics_for_raft = torch.stack([fx, fy, cx, cy], dim=1)
            
        # else:  # Single view case
        #     num_views, batch_size = 1, intrinsics.shape[0]
        #     depth1 = self.create_depth_map_from_reference_points_multiview(
        #         ref_points_0, intrinsics.unsqueeze(0).unsqueeze(0), H, W, num_views)
        #     depth2 = self.create_depth_map_from_reference_points_multiview(
        #         ref_points_1, intrinsics.unsqueeze(0).unsqueeze(0), H, W, num_views)
        # intrinsics_for_raft = intrinsics
        
        print(f"Debug - Created depth maps: depth1 {depth1.shape}, depth2 {depth2.shape}")
        print(f"Debug - RAFT intrinsics: {intrinsics_for_raft.shape}")
        
        # Calculate sampling to get exactly 64x64 from input size
        H, W = depth1.shape[1], depth1.shape[2]  # Should be 512, 960
        h_step = H // 64  # 512 // 64 = 8
        w_step = W // 64  # 960 // 64 = 15
        
        # Scale intrinsics according to actual downsampling ratios
        # fx, fy need to be scaled by the downsampling factors
        intrinsics_r8 = intrinsics_for_raft.clone()
        intrinsics_r8[:, 0] = intrinsics_r8[:, 0] / w_step  # fx scaling  
        intrinsics_r8[:, 1] = intrinsics_r8[:, 1] / h_step  # fy scaling
        intrinsics_r8[:, 2] = intrinsics_r8[:, 2] / w_step  # cx scaling
        intrinsics_r8[:, 3] = intrinsics_r8[:, 3] / h_step  # cy scaling
        
        # Downsample depth maps to match feature dimensions (64x64) and ensure float32
        # Original was depth1[:, 3::8, 3::8] which gave 64x120
        # We need to adjust to get 64x64 to match DQ transformer feature dimensions
        
        # Sample to get 64x64 dimensions
        depth1_r8 = depth1[:, ::h_step, ::w_step].float()  # [10, 64, 64]
        depth2_r8 = depth2[:, ::h_step, ::w_step].float()  # [10, 64, 64]
        

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
            
            corr = corr_fn(coords1_for_corr)
            flow = coords1 - coords0

            dz = zinv.unsqueeze(-1) - zinv_proj
            twist = Ts.log()

            net, mask, ae, delta, weight = \
                self.update_block(net, inp, corr, flow, dz, twist)

            target = coords1_xyz.permute(0,3,1,2) + delta
            target = target.contiguous()

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

        Ts_up = se3_field.upsample_se3(Ts, mask)
        return Ts_up

