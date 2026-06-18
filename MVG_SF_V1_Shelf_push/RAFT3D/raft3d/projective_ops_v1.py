import torch
import torch.nn.functional as F

from .sampler_ops import *

MIN_DEPTH = 0.05

def project(Xs, intrinsics):
    """ Pinhole camera projection """
    X, Y, Z = Xs.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[:,None,None].unbind(dim=-1)

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1.0 / Z

    coords = torch.stack([x, y, d], dim=-1)
    return coords

def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape[-2:]
    print(f"DEBUG inv_project: depths.shape = {depths.shape}")
    print(f"DEBUG inv_project: intrinsics.shape = {intrinsics.shape}")
    print(f"DEBUG inv_project: intrinsics sample = {intrinsics[0]}")
    
    # Add debug for broadcasting
    print(f"DEBUG inv_project: intrinsics[:,None,None].shape = {intrinsics[:,None,None].shape}")
    
    fx, fy, cx, cy = \
        intrinsics[:,None,None].unbind(dim=-1)
    
    print(f"DEBUG inv_project: fx.shape = {fx.shape}")
    print(f"DEBUG inv_project: ht={ht}, wd={wd}")

    y, x = torch.meshgrid(
        torch.arange(ht).to(depths.device).float(), 
        torch.arange(wd).to(depths.device).float())
    
    print(f"DEBUG inv_project: x.shape = {x.shape}")
    print(f"DEBUG inv_project: computing X = depths * ((x - cx) / fx)")
    print(f"DEBUG inv_project: x.shape = {x.shape}, cx.shape = {cx.shape}, fx.shape = {fx.shape}")
    
    # Remove channel dimension from depth if present [B, 1, H, W] -> [B, H, W] 
    if depths.ndim == 4 and depths.shape[1] == 1:
        depths = depths.squeeze(1)
        print(f"DEBUG inv_project: depths after squeeze = {depths.shape}")

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths
    
    print(f"DEBUG inv_project: X.shape = {X.shape}, Y.shape = {Y.shape}, Z.shape = {Z.shape}")

    return torch.stack([X, Y, Z], dim=-1)

def projective_transform(Ts, depth, intrinsics):
    """ Project points from I1 to I2 """
    
    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0
    x1 = project(X1, intrinsics)

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH)
    return x1, valid.float()

def induced_flow(Ts, depth, intrinsics):
    """ Compute 2d and 3d flow fields """

    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0

    x0 = project(X0, intrinsics)
    x1 = project(X1, intrinsics)

    flow2d = x1 - x0
    flow3d = X1 - X0

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH)
    return flow2d, flow3d, valid.float()


def backproject_flow3d(flow2d, depth0, depth1, intrinsics):
    """ compute 3D flow from 2D flow + depth change """

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = \
        intrinsics[None].unbind(dim=-1)
    
    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(), 
        torch.arange(wd).to(depth0.device).float())

    x1 = x0 + flow2d[...,0]
    y1 = y0 + flow2d[...,1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1

    flow3d = torch.stack([X1-X0, Y1-Y0, Z1-Z0], dim=-1)
    return flow3d

