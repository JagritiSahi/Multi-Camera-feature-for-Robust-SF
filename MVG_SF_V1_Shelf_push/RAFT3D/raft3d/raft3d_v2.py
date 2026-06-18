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
    def features_and_correlation(self, image1, image2 , meta=None):
        # extract features and build correlation volume
        # image1 = image[0][0]
        # image2 = image[1][0]
        print("(10)model input===>" , image1.shape , image2.shape)
        half = image1.shape[0] // 2
        image1_t0 = image1[:half]
        image2_t1 = image2[half]
        image_for_dq = [image1_t0, image2_t1]
        print("(11)model input===>", image_for_dq[0].shape)
        # fmap1, fmap2 = self.fnet([image1, image2])
        print("image===>" , len(image_for_dq))
        out , loss_dict = self.dq_model(image_for_dq, meta)        #*********************************
        fmap1 = out['hs_aligned_0']
        fmap2 = out['hs_aligned_1']
        # reference_points_0 = out['reference_points0']  # [B, N, 3] - 3D reference points frame 0
        # reference_points_1 = out['reference_points1']

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # extract context features using Resnet50
        net_inp = self.cnet(image1)
        net, inp = net_inp.split([128, 128*3], dim=1)

        net = torch.tanh(net)
        inp = torch.relu(inp)

        return corr_fn, net, inp

    def forward(self, image1, image2, depth1, depth2, intrinsics, iters=12, train_mode=False, meta=None):
    # def forward(self, image , meta ,depth1, depth2, intrinsics, iters=12, train_mode=False):
    # def forward(self, image , meta, intrinsics, iters=12, train_mode=False):
        """ Estimate optical flow between pair of frames """

        # image1 = image[0][0]
        # image2 = image[1][0]
        print("(8)model input===>" , image1.shape , image2.shape)
        Ts, coords0 = self.initializer(image1)
        print("(9)model input===>" , image1.shape , image2.shape)
        corr_fn, net, inp = self.features_and_correlation(image1, image2, meta=meta)

        # intrinsics and depth at 1/8 resolution
        intrinsics_r8 = intrinsics / 8.0
        # out , loss_dict = self.dq_model(image, meta) 
        # depth1 = out['reference_points0'][:, :, 2]  # [B, N, 3] - 3D reference points frame 0
        # depth2 = out['reference_points1'][:, :, 2]
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

