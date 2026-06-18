# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import json
import logging
import copy
import os
from collections import OrderedDict

from RAFT3D.data_readers import frame_utils
from lib.dataset.JointsDataset import JointsDataset
from lib.utils.transforms import projectPoints
import torch

logger = logging.getLogger(__name__)

# Campus_Seq1 uses 14 joints, mapping to Panoptic's 15 joints
# Campus_Seq1 joint order (14 joints):
# 0: Right-Ankle, 1: Right-Knee, 2: Right-Hip, 3: Left-Hip, 4: Left-Knee, 5: Left-Ankle,
# 6: Right-Wrist, 7: Right-Elbow, 8: Right-Shoulder, 9: Left-Shoulder, 10: Left-Elbow, 11: Left-Wrist,
# 12: Bottom-Head, 13: Top-Head

# Panoptic joint order (15 joints):
# 0: neck, 1: nose, 2: mid-hip, 3: l-shoulder, 4: l-elbow, 5: l-wrist,
# 6: l-hip, 7: l-knee, 8: l-ankle, 9: r-shoulder, 10: r-elbow, 11: r-wrist,
# 12: r-hip, 13: r-knee, 14: r-ankle

# Mapping from Campus_Seq1 (14) to Panoptic (15)
# We'll add mid-hip as average of left and right hip
CAMPUS_SEQ1_TO_PANOPTIC = {
    0: 14,   # Right-Ankle -> r-ankle
    1: 13,   # Right-Knee -> r-knee
    2: 12,   # Right-Hip -> r-hip
    3: 6,    # Left-Hip -> l-hip
    4: 7,    # Left-Knee -> l-knee
    5: 8,    # Left-Ankle -> l-ankle
    6: 11,   # Right-Wrist -> r-wrist
    7: 10,   # Right-Elbow -> r-elbow
    8: 9,    # Right-Shoulder -> r-shoulder
    9: 3,    # Left-Shoulder -> l-shoulder
    10: 4,   # Left-Elbow -> l-elbow
    11: 5,   # Left-Wrist -> l-wrist
    12: 1,   # Bottom-Head -> nose (approximation)
    13: 0,   # Top-Head -> neck (approximation)
}

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]


class CampusSeq1(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        
        # Override dataset_root to handle absolute paths directly
        if osp.isabs(cfg.DATASET.ROOT):
            # Use absolute path directly
            self.dataset_root_override = cfg.DATASET.ROOT
        else:
            # Use relative path handling from parent class
            this_dir = os.path.dirname(__file__)
            dataset_root = os.path.join(this_dir, '../..', cfg.DATASET.ROOT)
            self.dataset_root_override = os.path.abspath(dataset_root)
        
        super().__init__(cfg, image_set, is_train, transform)
        
        # Override dataset_root with our computed path
        self.dataset_root = self.dataset_root_override
        
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)  # 15 joints
        
        # Get camera list - check config or use default
        # Try to get from config, but handle if not present
        try:
            if hasattr(cfg.DATASET, 'CAM_LIST') and cfg.DATASET.CAM_LIST is not None:
                self.cam_list = cfg.DATASET.CAM_LIST
            else:
                self.cam_list = ['Camera0', 'Camera1', 'Camera2']
        except:
            self.cam_list = ['Camera0', 'Camera1', 'Camera2']
        self.num_views = len(self.cam_list)
        
        # Load database
        self.db = self._get_db()
        self.db_size = len(self.db)
        
        logger.info(f'CampusSeq1 dataset loaded: {self.db_size} entries, {self.num_views} views')

    def _get_cam(self):
        """Load and convert camera calibration from Tw matrix to R and t format"""
        # Handle both absolute and relative paths
        if osp.isabs(self.dataset_root):
            cam_file = osp.join(self.dataset_root, "calibration.json")
        else:
            cam_file = osp.join(self.dataset_root, "calibration.json")
        with open(cam_file) as cfile:
            calib_data = json.load(cfile)
        
        cameras = {}
        
        # Coordinate transformation matrix (similar to Panoptic)
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        
        for cam_name in self.cam_list:
            if cam_name not in calib_data.get('cameras', {}):
                logger.warning(f"Camera {cam_name} not found in calibration file")
                continue
                
            cam_data = calib_data['cameras'][cam_name]
            
            # Extract K matrix
            K = np.array(cam_data['K'])
            
            # Extract Tw (4x4 transform matrix) and convert to R and t
            Tw = np.array(cam_data['Tw'])
            R_world_to_cam = Tw[:3, :3]  # Rotation matrix
            t_world_to_cam = Tw[:3, 3:4]  # Translation vector (3x1)
            
            # Apply coordinate transformation
            R = R_world_to_cam.dot(M)
            
            # Extract distortion coefficients
            dist_coeffs = np.array(cam_data.get('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0]))
            # Ensure we have 5 distortion coefficients
            if len(dist_coeffs) < 5:
                dist_coeffs = np.pad(dist_coeffs, (0, 5 - len(dist_coeffs)), 'constant')
            
            # Store camera parameters in expected format
            cameras[cam_name] = {
                'K': K,
                'R': R,
                't': t_world_to_cam,
                'distCoef': dist_coeffs,
                'fx': K[0, 0],
                'fy': K[1, 1],
                'cx': K[0, 2],
                'cy': K[1, 2],
                'k': dist_coeffs[[0, 1, 4]].reshape(3, 1),
                'p': dist_coeffs[[2, 3]].reshape(2, 1),
            }
        
        return cameras

    def _map_14_to_15_joints(self, pose_14):
        """
        Map 14-joint Campus_Seq1 pose to 15-joint Panoptic format
        pose_14: numpy array of shape (14, 3)
        Returns: numpy array of shape (15, 3)
        """
        pose_15 = np.zeros((15, 3))
        
        # Map existing joints
        for campus_idx, panoptic_idx in CAMPUS_SEQ1_TO_PANOPTIC.items():
            pose_15[panoptic_idx] = pose_14[campus_idx]
        
        # Add mid-hip as average of left and right hip
        # Campus_Seq1: left-hip=3, right-hip=2
        # Panoptic: mid-hip=2
        left_hip = pose_14[3]  # Left-Hip
        right_hip = pose_14[2]  # Right-Hip
        pose_15[2] = (left_hip + right_hip) / 2.0  # mid-hip
        
        return pose_15

    def _get_db(self):
        """Load database from annotation files"""
        width = 360  # Default image width (should match actual images)
        height = 288  # Default image height (should match actual images)
        
        # Try to get image dimensions from annotation_2d.json
        annotation_2d_file = osp.join(self.dataset_root, "annotation_2d.json")
        if osp.exists(annotation_2d_file):
            with open(annotation_2d_file) as f:
                ann_2d_data = json.load(f)
                if 'image_wh' in ann_2d_data:
                    width, height = ann_2d_data['image_wh']
        
        db = []
        cameras = self._get_cam()
        
        # Load 3D annotations
        annotation_3d_file = osp.join(self.dataset_root, "annotation_3d.json")
        if not osp.exists(annotation_3d_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_3d_file}")
        with open(annotation_3d_file) as f:
            ann_3d_data = json.load(f)
        
        # Load 2D annotations for reference (optional)
        ann_2d_frames = {}
        annotation_2d_file = osp.join(self.dataset_root, "annotation_2d.json")
        if osp.exists(annotation_2d_file):
            with open(annotation_2d_file) as f:
                ann_2d_data = json.load(f)
                ann_2d_frames = ann_2d_data.get('frames', {})
        
        # Process each frame
        for frame_idx, frame_data in enumerate(ann_3d_data):
            timestamp = frame_data.get('timestamp', frame_idx * 0.04)  # Default 25fps
            poses_3d = frame_data.get('poses', [])
            
            if len(poses_3d) == 0:
                continue
            
            # Process each camera view
            for cam_name, cam in cameras.items():
                # Construct image path
                # Format: frames/Camera{N}/timestamp.jpg (e.g., 0000.000.jpg)
                # Round timestamp to nearest 0.04-second interval to match actual filenames
                # Files are stored at 25fps intervals (0.04 seconds apart)
                frame_interval = 0.04
                rounded_timestamp = round(timestamp / frame_interval) * frame_interval
                # Format the timestamp to match file naming: 0000.000 format
                # Split into integer and decimal parts
                int_part = int(rounded_timestamp)
                # Use round here as well to avoid 0.679999 → 679
                dec_part = int(round((rounded_timestamp - int_part) * 1000))
                # Handle edge case where rounding gives 1000
                if dec_part == 1000:
                    int_part += 1
                    dec_part = 0
                timestamp_str = f"{int_part:04d}.{dec_part:03d}"  # Format: 0000.000
                image_path = osp.join("frames", cam_name, f"{timestamp_str}.jpg")
                
                # Get 2D annotation for this camera if available
                frame_key_2d = f"{cam_name}/{timestamp:.3f}.jpg"
                ann_2d_frame = ann_2d_frames.get(frame_key_2d, {})
                
                all_poses_3d = []
                all_poses_vis_3d = []
                all_poses = []
                all_poses_vis = []
                
                # Process each person's pose
                for pose_data in poses_3d:
                    points_3d = np.array(pose_data['points_3d'])  # Shape: (14, 3)
                    
                    if len(points_3d) != 14:
                        logger.warning(f"Unexpected number of joints: {len(points_3d)}")
                        continue
                    
                    # Map to 15-joint format
                    pose_3d_15 = self._map_14_to_15_joints(points_3d)
                    
                    # Create visibility array (all joints visible initially)
                    joints_vis_3d = np.ones((15, 3))
                    
                    # Project to 2D
                    pose_3d_transposed = pose_3d_15.transpose()  # (3, 15)
                    pose_2d = projectPoints(
                        pose_3d_transposed,
                        cam['K'],
                        cam['R'],
                        cam['t'],
                        cam['distCoef']
                    ).transpose()[:, :2]  # (15, 2)
                    
                    # Check visibility based on image bounds
                    x_check = np.bitwise_and(pose_2d[:, 0] >= 0,
                                           pose_2d[:, 0] <= width - 1)
                    y_check = np.bitwise_and(pose_2d[:, 1] >= 0,
                                           pose_2d[:, 1] <= height - 1)
                    check = np.bitwise_and(x_check, y_check)
                    
                    # Update visibility
                    joints_vis_3d[np.logical_not(check), :] = 0
                    joints_vis_2d = np.repeat(
                        np.reshape(check.astype(float), (-1, 1)), 2, axis=1)
                    
                    # Check if root joint (mid-hip) is visible
                    if joints_vis_3d[2, 0] < 0.1:
                        continue  # Skip if root joint not visible
                    
                    all_poses_3d.append(pose_3d_15)
                    all_poses_vis_3d.append(joints_vis_3d)
                    all_poses.append(pose_2d)
                    all_poses_vis.append(joints_vis_2d)
                
                if len(all_poses_3d) > 0:
                    # Prepare camera parameters in expected format
                    our_cam = {}
                    our_cam['R'] = cam['R']
                    our_cam['T'] = -np.dot(cam['R'].T, cam['t']) * 10.0  # cm to mm
                    our_cam['standard_T'] = cam['t'] * 10.0
                    our_cam['fx'] = cam['fx']
                    our_cam['fy'] = cam['fy']
                    our_cam['cx'] = cam['cx']
                    our_cam['cy'] = cam['cy']
                    our_cam['k'] = cam['k']
                    our_cam['p'] = cam['p']
                    
                    db.append({
                        'key': f"{cam_name}_{frame_idx}_{timestamp:.3f}",
                        'image': osp.join(self.dataset_root, image_path),
                        'joints_3d': all_poses_3d,
                        'joints_3d_vis': all_poses_vis_3d,
                        'joints_2d': all_poses,
                        'joints_2d_vis': all_poses_vis,
                        'camera': our_cam,
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                    })
        
        return db

    def __getitem__(self, idx):
        """Return 11-tuple: images, meta, flow, disparity, and scene flow."""
        inputs, input_t1, meta, meta_t1 = [], [], [], []
        flows = []
        valids = []
        disps = []
        disps_t1 = []
        disps_change = []
        sceneflows = []
        sceneflow_valids = []
        
        for k in range(self.num_views):
            # We want the same camera at time t and t+1.
            # db is arranged as [frame0_cam0, frame0_cam1, ..., frame1_cam0, frame1_cam1, ...]
            # base index for this timestamp block
            base = self.num_views * idx
            curr_pos = base + k
            next_pos = curr_pos + self.num_views
            # if next_pos goes beyond dataset, duplicate current (same behavior as parent)
            if next_pos >= self.db_size:
                next_pos = curr_pos

            # Call parent to get processed data for the two absolute db indices.
            # parent returns (inputs, input_t1, meta, meta_t1) where inputs corresponds
            # to the requested db index. We only need the first returned image/meta
            # for each call (they correspond to that db record).
            i_curr, _, m_curr, _ = super().__getitem__(curr_pos)
            i_next, _, m_next, _ = super().__getitem__(next_pos)

            i, i_t1, m, m_t1 = i_curr, i_next, m_curr, m_next
            
            # Load optical flow
            flow_path = m['image'].replace('hdImgs', 'hdFlow')
            if os.path.exists(flow_path):
                flow = frame_utils.read_gen(flow_path)
                flow = np.array(flow).astype(np.float32)
                u = (flow[..., 0] - 128.0) / 64.0
                v = (flow[..., 1] - 128.0) / 64.0
                flow = np.stack([u, v], axis=-1)
                flow = torch.from_numpy(flow).permute(2, 0, 1).float()
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
                valid = valid.float()
            else:
                flow = torch.zeros((2, i.shape[1], i.shape[2]))
                valid = torch.zeros((i.shape[1], i.shape[2]))
            
            # Load disparity at time t
            disp_path = m['image'].replace('hdImgs', 'hdDepths').replace('.jpg', '.npy')
            if os.path.exists(disp_path):
                disp = np.load(disp_path)
                disp = np.array(disp).astype(np.float32) / 256.0
                disp = torch.from_numpy(disp).unsqueeze(0).float()
            else:
                disp = torch.zeros((1, i.shape[1], i.shape[2]))
            
            # Load disparity at time t+1
            disp_path_t1 = m_t1['image'].replace('hdImgs', 'hdDepths').replace('.jpg', '.npy')
            if os.path.exists(disp_path_t1):
                disp_t1 = np.load(disp_path_t1)
                disp_t1 = np.array(disp_t1).astype(np.float32) / 256.0
                disp_t1 = torch.from_numpy(disp_t1).unsqueeze(0).float()
            else:
                disp_t1 = torch.zeros((1, i.shape[1], i.shape[2]))
            
            # Compute or load disparity change
            disp_change_path = m['image'].replace('hdImgs', 'hdDisparityChange').replace('.jpg', '.png')
            if os.path.exists(disp_change_path):
                # Load precomputed disparity change
                disp_change = frame_utils.read_gen(disp_change_path)
                disp_change = np.array(disp_change).astype(np.float32) / 256.0
                disp_change = torch.from_numpy(disp_change).unsqueeze(0).float()
            else:
                disp_change = disp_t1 - disp
            
            # Load scene flow
            sceneflow_path = m['image'].replace('hdImgs', 'hdSceneflow').replace('.jpg', '.npy')
            if os.path.exists(sceneflow_path):
                sceneflow = frame_utils.read_gen(sceneflow_path)
                sceneflow = np.array(sceneflow).astype(np.float32)
                
                if len(sceneflow.shape) == 3 and sceneflow.shape[2] >= 3:
                    sceneflow = torch.from_numpy(sceneflow[..., :3]).permute(2, 0, 1).float()
                    sceneflow_valid = (sceneflow[0].abs() < 1000) & \
                                      (sceneflow[1].abs() < 1000) & \
                                      (sceneflow[2].abs() < 1000) & \
                                      torch.isfinite(sceneflow[0]) & \
                                      torch.isfinite(sceneflow[1]) & \
                                      torch.isfinite(sceneflow[2])
                    sceneflow_valid = sceneflow_valid.float()
                else:
                    sceneflow = torch.zeros((3, i.shape[1], i.shape[2]))
                    sceneflow_valid = torch.zeros((i.shape[1], i.shape[2]))
            else:
                sceneflow = torch.zeros((3, i.shape[1], i.shape[2]))
                sceneflow_valid = torch.zeros((i.shape[1], i.shape[2]))
            
            inputs.append(i)
            input_t1.append(i_t1)
            meta.append(m)
            meta_t1.append(m_t1)
            flows.append(flow)
            valids.append(valid)
            disps.append(disp)
            disps_t1.append(disp_t1)
            disps_change.append(disp_change)
            sceneflows.append(sceneflow)
            sceneflow_valids.append(sceneflow_valid)

        return inputs, input_t1, meta, meta_t1, flows, valids, disps, disps_t1, disps_change, sceneflows, sceneflow_valids

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, preds, method='score_sort'):
        """Evaluate predictions against ground truth"""
        eval_list = []
        gt_num = self.db_size // self.num_views
        assert len(preds) == gt_num, f'number mismatch: preds={len(preds)}, gt_num={gt_num}'
        
        total_gt = 0
        
        for i in range(gt_num):
            index = self.num_views * i
            if index >= len(self.db):
                continue
                
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']
            
            if len(joints_3d) == 0:
                continue
            
            pred = preds[i].copy()
            
            if method == 'mpjpe_sort':
                # No filtering with classification
                gt_id_list = []
                for pose in pred:
                    mpjpes = []
                    for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                        vis = gt_vis[:, 0] > 0
                        mpjpe = np.mean(np.sqrt(
                            np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                        mpjpes.append(mpjpe)
                    min_gt = np.argmin(mpjpes)
                    min_mpjpe = np.min(mpjpes)
                    score = pose[0, 4] if pose.shape[1] > 4 else 1.0
                    
                    gt_id = int(total_gt + min_gt)
                    if gt_id not in gt_id_list:
                        eval_list.append({
                            "mpjpe": float(min_mpjpe),
                            "score": float(score),
                            "gt_id": gt_id
                        })
                        gt_id_list.append(gt_id)
            else:
                # Filter with classification threshold
                pred = pred[pred[:, 0, 3] >= 0]
                for pose in pred:
                    mpjpes = []
                    for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                        vis = gt_vis[:, 0] > 0
                        mpjpe = np.mean(np.sqrt(
                            np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                        mpjpes.append(mpjpe)
                    min_gt = np.argmin(mpjpes)
                    min_mpjpe = np.min(mpjpes)
                    score = pose[0, 4] if pose.shape[1] > 4 else 1.0
                    eval_list.append({
                        "mpjpe": float(min_mpjpe),
                        "score": float(score),
                        "gt_id": int(total_gt + min_gt)
                    })
            
            total_gt += len(joints_3d)
        
        # Calculate metrics
        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t, method=method)
            aps.append(ap)
            recs.append(rec)
        
        mpjpe = self._eval_list_to_mpjpe(eval_list, method=method)
        recall500 = self._eval_list_to_recall(eval_list, total_gt)
        
        return aps, recs, mpjpe, recall500

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold, method='score_sort'):
        """Calculate Average Precision at given threshold"""
        if method == 'score_sort':
            eval_list.sort(key=lambda k: k["score"], reverse=True)
        else:
            eval_list.sort(key=lambda k: k["mpjpe"])
        
        tp = np.zeros(len(eval_list))
        fp = np.zeros(len(eval_list))
        gt_detected = set()
        
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_detected:
                tp[i] = 1
                gt_detected.add(item["gt_id"])
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        rec = tp_cumsum / (total_gt + 1e-8)
        prec = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        ap = np.trapz(prec, rec)
        return ap, rec[-1] if len(rec) > 0 else 0.0

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, method='score_sort'):
        """Calculate mean MPJPE"""
        if len(eval_list) == 0:
            return 0.0
        mpjpes = [item["mpjpe"] for item in eval_list]
        return np.mean(mpjpes)

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        """Calculate recall at threshold"""
        if total_gt == 0:
            return 0.0
        detected = len([item for item in eval_list if item["mpjpe"] < threshold])
        return detected / total_gt
