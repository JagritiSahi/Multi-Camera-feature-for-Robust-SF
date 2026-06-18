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
        self.dq_model.eval()
        self.dq_model.to('cuda')

    def create_depth_map_from_reference_points(self,reference_points, intrinsics, image_height, image_width):
        """
        Convert sparse 3D reference points to dense depth maps
        
        Args:
            reference_points: [B, N*joints, 3] - 3D points in camera coordinates
            intrinsics: [B, 4] - camera intrinsics [fx, fy, cx, cy]
            image_height, image_width: target depth map dimensions
        
        Returns:
            depth_map: [B, H, W] - dense depth map
        """
        batch_size = reference_points.shape[0]
        device = reference_points.device
        
        # Initialize depth map
        depth_map = torch.zeros(batch_size, image_height, image_width, device=device)
        
        for b in range(batch_size):
            points_3d = reference_points[b]  # [N*joints, 3]
            intrinsic = intrinsics[b]  # [4]
            
            # Project 3D points to 2D image coordinates
            X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            
            # Project to pixel coordinates
            u = (fx * X / Z + cx).round().long()
            v = (fy * Y / Z + cy).round().long()
            
            # Filter valid projections
            valid_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height) & (Z > 0)
            valid_u = u[valid_mask]
            valid_v = v[valid_mask]
            valid_z = Z[valid_mask]
            
            # Fill depth map at projected locations
            depth_map[b, valid_v, valid_u] = valid_z
        
        # Interpolate to fill gaps (optional)
        # You can use inpainting, nearest neighbor, or other interpolation methods
        depth_map = self.interpolate_depth_gaps(depth_map)
    
        return depth_map



    def interpolate_depth_gaps(self , depth_map):
        """Fill gaps in sparse depth map using interpolation"""
        # Simple approach: use nearest neighbor for empty pixels
        # More sophisticated: use RBF, bilinear, or neural inpainting
        
        for b in range(depth_map.shape[0]):
            depth_slice = depth_map[b]
            mask = depth_slice > 0
            
            if mask.sum() > 0:
                # Get coordinates of valid depth values
                valid_coords = torch.nonzero(mask, as_tuple=False).float()
                valid_depths = depth_slice[mask]
                
                # Create grid for all pixels
                h, w = depth_slice.shape
                y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w))
                all_coords = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1).float().to(depth_map.device)
                
                # Simple nearest neighbor interpolation
                if len(valid_coords) > 0:
                    distances = torch.cdist(all_coords, valid_coords)
                    nearest_indices = distances.argmin(dim=1)
                    interpolated_depths = valid_depths[nearest_indices]
                    depth_map[b] = interpolated_depths.view(h, w)
        
        return depth_map

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

        y0, x0 = torch.meshgrid(torch.arange(ht//8), torch.arange(wd//8))
        coords0 = torch.stack([x0, y0], dim=-1).float()
        coords0 = coords0[None].repeat(batch_size, 1, 1, 1).to(device)

        Ts = SE3.Identity(batch_size, ht//8, wd//8, device=device)
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
        # half = image1.shape[0] // 2
        # image1_t0 = image1[:half]
        # image2_t1 = image2[half]
        # image_for_dq = [image1_t0, image2_t1]
        # print("(11)model input===>", image_for_dq[0].shape)
        # fmap1, fmap2 = self.fnet([image1, image2])
        # print("image===>" , len(image_for_dq))
        dq_output = self.dq_model(image_for_dq, meta)        #*********************************
        
        # Handle different return types from DQ model
        if isinstance(dq_output, tuple):
            out, loss_dict = dq_output
        else:
            out = dq_output
            loss_dict = None
            
        # Use the 4D fixed features instead of 3D aligned features for RAFT compatibility
        fmap1 = out['attn_feature_views_0']  # Should be (B, C, H, W)
        fmap2 = out['attn_feature_views_1']  # Should be (B, C, H, W)
        print(f"DQ feature shapes: fmap1={fmap1.shape}, fmap2={fmap2.shape}")
        
        # reference_points_0 = out['reference_points0']  # [B, N, 3] - 3D reference points frame 0
        # reference_points_1 = out['reference_points1']

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # extract context features using Resnet50
        net_inp = self.cnet(image1)
        net, inp = net_inp.split([128, 128*3], dim=1)

        net = torch.tanh(net)
        inp = torch.relu(inp)

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

        # intrinsics and depth at 1/8 resolution
        intrinsics_r8 = intrinsics / 8.0
        # out , loss_dict = self.dq_model([image1, image2], meta) 
        dq_output = self.dq_model(image_for_dq, meta) 
        out, loss_dict = dq_output if isinstance(dq_output, tuple) else (dq_output, None)

        # Extract reference points
        ref_points_0 = out['reference_points0']  # [B, N*joints, 3]
        ref_points_1 = out['reference_points1']  # [B, N*joints, 3]
        
        # Convert to dense depth maps
        _, _, H, W = image1.shape
        depth1 = self.create_depth_map_from_reference_points(ref_points_0, intrinsics, H, W)
        depth2 = self.create_depth_map_from_reference_points(ref_points_1, intrinsics, H, W)

        depth1 = out['reference_points0'][:, :, 2]  # [B, N, 3] - 3D reference points frame 0
        depth2 = out['reference_points1'][:, :, 2]
        depth1_r8 = depth1[:,3::8,3::8]
        depth2_r8 = depth2[:,3::8,3::8]
        

        flow_est_list = []
        flow_rev_list = []

        for itr in range(iters):
            Ts = Ts.detach()

            coords1_xyz, _ = pops.projective_transform(Ts, depth1_r8, intrinsics_r8)
            
            coords1, zinv_proj = coords1_xyz.split([2,1], dim=-1)
            zinv, _ = depth_sampler(1.0/depth2_r8, coords1)

            corr = corr_fn(coords1.permute(0,3,1,2).contiguous())
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

