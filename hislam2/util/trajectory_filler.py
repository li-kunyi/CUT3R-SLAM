import torch
import lietorch
from lietorch import SE3
import os
import torch
import lietorch
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation
from util.utils import get_depth_normal, viz_map, umeyama_alignment, weighted_align_point_maps
from util.utils import viz_pcd

class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses """

    def __init__(self, slam, device="cuda:0"):
        
        # split net modules
        self.model = slam.model
        self.mapper = slam.mapper

        self.count = 0
        self.keyframes = slam.keyframes
        self.device = device

        self.downsample_ratio = slam.downsample_ratio
        self.output_dir = slam.output_dir

    def _PnP(self, points_2d, K, pts, mask):
        if mask is not None:
            pts = pts[mask]
            points_2d = points_2d[mask]

        pts = pts.reshape(-1, 3)
        points_2d = points_2d.reshape(-1, 2)

        _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(
                pts.astype(np.float32), 
                points_2d.astype(np.float32), 
                K.astype(np.float32), 
                np.zeros(4).astype(np.float32))
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            # Extrinsic parameters (4x4 matrix)
        w2c = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        w2c = np.vstack((w2c, [0, 0, 0, 1]))
        c2w = np.linalg.inv(w2c)

        r = Rotation.from_matrix(c2w[:3, :3])
        quat = torch.from_numpy(r.as_quat())
        trans = torch.from_numpy(c2w[:3, 3])

        w2c = torch.from_numpy(w2c).to(torch.float32)
        
        return w2c, quat, trans
    

    @torch.no_grad()
    def fill(self, tstamp0, tstamp1, images, img0=None, pose0=None):
        """ fill operator """

        B, C, H, W = img0.shape
        new_poses = []
        prev_pose = pose0
        pbar = tqdm(range(tstamp0+1, tstamp1), desc="Pose Filling")
        for i in range(tstamp0+1, tstamp1):
            img_i = images[i]
            
            # gaussian refinement
            with torch.enable_grad():
                pose = self.mapper.pose_estimator(prev_pose, img_i.squeeze(0), i)

            # pts_np = pointmap.detach().cpu().reshape(-1, 3).numpy()
            # img_np = img_i.squeeze(0).permute(1, 2, 0).detach().cpu().reshape(-1, 3).numpy() / 255.0
            # viz_pcd(pts_np, img_np, os.path.join(self.output_dir, "fill"), name=f"frame_{i}.ply")

            new_poses.append(pose)
            prev_pose = pose

            pbar.set_postfix(frame=i)
            pbar.update()

        return torch.stack(new_poses, dim=0)


    @torch.no_grad()
    def run(self, image_stream):
        """ fill in poses of non-keyframe images """

        # store all camera poses
        pose_list = []
        
        kf_num = self.keyframes.counter.value - 1
        
        for i in range(kf_num - 1):          
            tstamp0 = int(self.keyframes.tstamp[i].cpu().item())
            pose0 = self.keyframes.pose[i]
            img0 = self.keyframes.image[i].unsqueeze(0)
            tstamp1 = int(self.keyframes.tstamp[i+1].cpu().item())

            new_pose = self.fill(tstamp0, tstamp1, image_stream, img0, pose0)
            pose_batch = torch.cat([pose0[None], new_pose], dim=0)
            pose_list.append(pose_batch)

        tstamp0 = int(self.keyframes.tstamp[kf_num - 1].cpu().item())
        pose0 = self.keyframes.pose[kf_num - 1]
        img0 = self.keyframes.image[kf_num - 1].unsqueeze(0)
        new_pose = self.fill(tstamp0, len(image_stream), image_stream, img0, pose0)
        pose_batch = torch.cat([pose0[None], new_pose], dim=0)
        pose_list.append(pose_batch)

        return torch.cat(pose_list, dim=0)
