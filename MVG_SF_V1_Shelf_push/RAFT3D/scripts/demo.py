import os
import sys

# Resolve symlinks to get the real script directory
script_dir = os.path.dirname(os.path.realpath(__file__))
# Add RAFT3D directory (parent) to sys.path for raft3d module imports
raft3d_root = os.path.abspath(os.path.join(script_dir, '..'))
# Add parent directory (MVG_SF_V1_Shelf) for lib imports
repo_root = os.path.abspath(os.path.join(raft3d_root, '..'))
# Insert in correct order: RAFT3D first, then parent
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if raft3d_root not in sys.path:
    sys.path.insert(0, raft3d_root)

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from lietorch import SE3
import raft3d.projective_ops as pops
from data_readers import frame_utils
from utils import show_image, normalize_image


DEPTH_SCALE = 0.2

def prepare_images_and_depths(image1, image2, depth1, depth2):
    """ padding, normalization, and scaling """

    image1 = F.pad(image1, [0,0,0,4], mode='replicate')
    image2 = F.pad(image2, [0,0,0,4], mode='replicate')
    depth1 = F.pad(depth1[:,None], [0,0,0,4], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,0,0,4], mode='replicate')[:,0]

    depth1 = (DEPTH_SCALE * depth1).float()
    depth2 = (DEPTH_SCALE * depth2).float()
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

    return image1, image2, depth1, depth2


def display(img, tau, phi):
    """ display se3 fields """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(img[:, :, ::-1] / 255.0)

    tau_img = np.clip(tau, -0.1, 0.1)
    tau_img = (tau_img + 0.1) / 0.2

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    plt.show()


@torch.no_grad()
def demo(args):
    import importlib
    # Build asset paths based on script location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    raft3d_root = os.path.abspath(os.path.join(script_dir, '..'))
    assets_dir = os.path.join(raft3d_root, 'assets')
    
    RAFT3D = importlib.import_module(args.network).RAFT3D
    model = torch.nn.DataParallel(RAFT3D(args))
    model.load_state_dict(torch.load(args.model), strict=False)

    model.eval()
    model.cuda()

    fx, fy, cx, cy = (1050.0, 1050.0, 480.0, 270.0)
    img1 = cv2.imread(os.path.join(assets_dir, 'image1.png'))
    img2 = cv2.imread(os.path.join(assets_dir, 'image2.png'))
    disp1 = frame_utils.read_gen(os.path.join(assets_dir, 'disp1.pfm'))
    disp2 = frame_utils.read_gen(os.path.join(assets_dir, 'disp2.pfm'))

    depth1 = torch.from_numpy(fx / disp1).float().cuda().unsqueeze(0)
    depth2 = torch.from_numpy(fx / disp2).float().cuda().unsqueeze(0)
    image1 = torch.from_numpy(img1).permute(2,0,1).float().cuda().unsqueeze(0)
    image2 = torch.from_numpy(img2).permute(2,0,1).float().cuda().unsqueeze(0)
    intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda().unsqueeze(0)

    image1, image2, depth1, depth2 = prepare_images_and_depths(image1, image2, depth1, depth2)
    Ts = model(image1, image2, depth1, depth2, intrinsics, iters=16)
    
    # compute 2d and 3d from from SE3 field (Ts)
    flow2d, flow3d, _ = pops.induced_flow(Ts, depth1, intrinsics)

    # extract rotational and translational components of Ts
    tau, phi = Ts.log().split([3,3], dim=-1)
    tau = tau[0].cpu().numpy()
    phi = phi[0].cpu().numpy()

    # undo depth scaling
    flow3d = flow3d / DEPTH_SCALE

    display(img1, tau, phi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft3d.pth', help='checkpoint to restore')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    args = parser.parse_args()

    demo(args)

    


