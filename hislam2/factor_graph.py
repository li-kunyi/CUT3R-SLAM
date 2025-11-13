import torch
import lietorch
import numpy as np

from tqdm import trange
# from lietorch import SE3
import torch.nn.functional as F
# from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from util.utils import pose_vec_to_matrix
from matplotlib.collections import LineCollection

class FactorGraph:
    def __init__(self, keyframes, device="cuda:0", max_factors=-1):
        self.keyframes = keyframes
        self.device = device
        self.max_factors = max_factors

        self.ii = torch.as_tensor([], dtype=torch.long, device=device)  # graph, from ii to jj
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)
        self.weight = torch.as_tensor([], dtype=torch.float32, device=device)  # strores overlap ratio of image i projected to image j
    

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)

    @torch.amp.autocast('cuda',enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors and remove:
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        self.ii = torch.cat([self.ii, ii], dim=0)
        self.jj = torch.cat([self.jj, jj], dim=0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], dim=0)
        # self.weight = torch.cat([self.weight, weight], dim=0)


    @torch.amp.autocast('cuda',enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """
        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]

    @torch.amp.autocast('cuda',enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """

        with self.keyframes.get_lock():
            self.keyframes.tstamp[ix] = self.keyframes.tstamp[ix+1]
            self.keyframes.image[ix] = self.keyframes.image[ix+1]
            self.keyframes.pose[ix] = self.keyframes.pose[ix+1]
            self.keyframes.intrinsic[ix] = self.keyframes.intrinsic[ix+1]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)


    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1), indexing='ij')
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])


    def find_potentially_overlapping_frames(
        self,
        current_pose,
        all_poses,
        dist_thresh=1.0,
    ):
        N = all_poses.shape[0]

        current_center = current_pose[:3, 3]
        centers = all_poses[:, :3, 3]
        dists = torch.norm(centers - current_center[None], dim=1)
        
        current_z = current_pose[:3, 2]
        z_vectors = all_poses[:, :3, 2]
        dot_prods = torch.sum(z_vectors * current_z[None], dim=1)
        norm_z = torch.norm(z_vectors, dim=1) * torch.norm(current_z)
        cos_sim = dot_prods / (norm_z + 1e-8)

        # Condition 1: close in space and looking in a similar direction (cos_sim > 0.9)
        cond1 = (dists < dist_thresh) & (cos_sim >= 0.9)
        
        # Condition 2: moderately similar viewing direction (0.8 < cos_sim <= 0.9)
        cond2 = (cos_sim >= 0.8) & (cos_sim < 0.9)

        final_mask = cond1 | cond2

        return torch.where(final_mask)[0]

    def add(self, current_idx, all_poses, all_pointmaps, current_pose, current_pointmap, K):

        current_center = current_pose[:3, 3]
        centers = all_poses[:, :3, 3]
        dists = torch.norm(centers - current_center[None], dim=1)
        
        current_z = current_pose[:3, 2]
        z_vectors = all_poses[:, :3, 2]
        dot_prods = torch.sum(z_vectors * current_z[None], dim=1)
        norm_z = torch.norm(z_vectors, dim=1) * torch.norm(current_z)
        cos_sim = dot_prods / (norm_z + 1e-8)

        # Condition 1: close in space and looking in a similar direction (cos_sim > 0.9)
        cond1 = (dists <= 1.0) #& (cos_sim >= 0.9)
        
        selected_c2w = all_poses[cond1]
        selected_idx = torch.where(cond1)[0]

        if selected_c2w.numel() > 0:
            overlap_ratios = self.cal_overlap_batch(current_pointmap, selected_c2w, K)  # reprojection overlap ratios
            mask = overlap_ratios > 0.3  # close views, should have large overlap          

            jj = selected_idx[mask]  # [M]
            if jj.numel() > 0:
                ii = torch.full_like(jj, current_idx, dtype=torch.long, device=self.device)
                self.add_factors(ii, jj)
                self.add_factors(jj, ii)

        # Condition 2: similar viewing direction (0.8 < cos_sim, 60 degree fov), but large distance difference
        # cond2 = (cos_sim >= 0.7) & (dists > 1.0) #& (dists < 5.0)
        # cond2 = cond2 #& (~cond1)
        cond2 = ~cond1

        selected_c2w = all_poses[cond2]
        selected_pointmaps = all_pointmaps[cond2]
        selected_idx = torch.where(cond2)[0]
        if selected_c2w.numel() > 0:
            overlap_ratios = self.cal_overlap_batch(current_pointmap, selected_c2w, K)  # reprojection overlap ratios
            # bi-directional overlap check
            overlap_ratios_a2c = self.cal_overlap_bi(selected_pointmaps, current_pose.unsqueeze(0), K).squeeze()  # [B, 1]
            # mask = (overlap_ratios + overlap_ratios_a2c) / 2 > 0.3
            mask = (overlap_ratios > 0.3) | (overlap_ratios_a2c > 0.3)

            jj = selected_idx[mask]  # [M]
            if jj.numel() > 0:
                ii = torch.full_like(jj, current_idx, dtype=torch.long, device=self.device)
                self.add_factors(ii, jj)
                self.add_factors(jj, ii)
        
        self.age += 1

    def update(self, t0, t1, K):
        
        H, W, _ = self.keyframes.submap_ds[0, 0].shape
        for i in range(t0 + 1, t1 + 1):
            sub_num = i // 5
            if i <= 3:
                continue
    
            if i > 3:
                all_pointmaps = self.keyframes.submap_ds[:sub_num + 1, :-1].reshape(-1, H, W, 3)
                all_pointmaps = (all_pointmaps[:i]).to(self.device)  
            else:
                all_pointmaps = self.keyframes.submap_ds[sub_num, :i-t0].to(self.device) 

            all_poses = self.keyframes.pose[:i].to(self.device)  # [i, 7]
            all_poses = all_poses.reshape(-1, 7)
            all_c2ws = pose_vec_to_matrix(all_poses)

            current_pose = self.keyframes.pose[i].to(self.device)  # [7]
            current_c2w = pose_vec_to_matrix(current_pose.unsqueeze(0))[0]  # [1, 4, 4]
            current_pts = self.keyframes.submap_ds[i//5][i%5].to(self.device) 

            mask = (self.ii == i) | (self.jj == i)
            self.rm_factors(mask)
            if i >= 3:
                self.add_neighborhood_factors(i-3, i+1, r=3)
            self.add(i, all_c2ws, all_pointmaps, current_c2w, current_pts, K)


    def cal_overlap(self, pointmap_i, T_j, K):
        H, W, _ = pointmap_i.shape
        points = pointmap_i.reshape(-1, 3)  # [N, 3]
        ones = torch.ones((points.shape[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=-1)  # [N, 4]

        # Transform world coordinates to target frame camera coordinates
        T_world_to_cam = torch.inverse(T_j)  # [4, 4]
        points_cam = (T_world_to_cam @ points_h.T).T[:, :3]  # [N, 3]

        # Project to the image plane
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2].clamp(min=1e-5)

        u = (K[0, 0] * x / z + K[0, 2]).round().long()
        v = (K[1, 1] * y / z + K[1, 2]).round().long()

        # Retain valid pixel ranges
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[valid]
        v = v[valid]
        # Calculate the ratio of valid pixels to total pixels
        overlap = valid.sum().float() / points.shape[0]

        return overlap
        
    def cal_overlap_batch(self, pointmap_i, T_j_batch, K):
        H, W, _ = pointmap_i.shape
        N = H * W
        B = T_j_batch.shape[0]

        points = pointmap_i.reshape(-1, 3)
        ones = torch.ones((N, 1), device=points.device)
        points_h = torch.cat([points, ones], dim=-1)  # [N, 4]
        points_h_batch = points_h.unsqueeze(0).expand(B, N, 4).float()  # [B, N, 4]

        T_world_to_cam_batch = torch.inverse(T_j_batch)  # [B, 4, 4]
        points_cam_batch = torch.bmm(points_h_batch, T_world_to_cam_batch.transpose(1, 2))[:, :, :3]  # [B, N, 3]

        x = points_cam_batch[:, :, 0]
        y = points_cam_batch[:, :, 1]
        z = points_cam_batch[:, :, 2].clamp(min=1e-5)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = (fx * x / z + cx).round().long()
        v = (fy * y / z + cy).round().long()

        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (points_cam_batch[:, :, 2] > 0)  # [B, N]
        overlap_ratios = valid.sum(dim=1).float() / N  # [B]

        return overlap_ratios  # [B]
    
    def cal_overlap_bi(self, pointmap_i, T_j_batch, K):
        # pointmap_i: [B1, H, W, 3]
        # T_j_batch: [B2, 4, 4]
        B1, H, W, _ = pointmap_i.shape
        B2 = T_j_batch.shape[0]
        N = H * W

        points = pointmap_i.reshape(B1, N, 3)  # [B1, N, 3]
        ones = torch.ones((B1, N, 1), device=points.device)
        points_h = torch.cat([points, ones], dim=-1)  # [B1, N, 4]

        # [B1, B2, N, 4]
        points_h = points_h.unsqueeze(1).expand(B1, B2, N, 4)
        T_j_batch = T_j_batch.unsqueeze(0).expand(B1, B2, 4, 4)
        T_world_to_cam_batch = torch.inverse(T_j_batch)  # [B1, B2, 4, 4]

        # [B1, B2, N, 3]
        points_cam_batch = torch.matmul(points_h, T_world_to_cam_batch.transpose(-1, -2))[..., :3]

        x = points_cam_batch[..., 0]
        y = points_cam_batch[..., 1]
        z = points_cam_batch[..., 2]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = (fx * x / z + cx).round().long()
        v = (fy * y / z + cy).round().long()

        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)  # [B1, B2, N]
        overlap_ratios = valid.sum(dim=2).float() / N  # [B1, B2]

        return overlap_ratios  # [B1, B2]

    
    def compute_feature_overlap_ratio(self, feat0, feat1, threshold=0.7):
        feat0 = F.normalize(feat0, dim=1)
        feat1 = F.normalize(feat1, dim=1)
        sim = torch.mm(feat0, feat1.T)
        max_sim, _ = sim.max(dim=1)
        matched = (max_sim > threshold).float()
        ratio = matched.mean().item()

        return ratio
    
    def compute_feature_overlap_batch(self, feat0, feat1_batch, threshold=0.7, return_item=False):
        # the 0th patch always similar, exclude it
        feat0 = F.normalize(feat0[1:], dim=1)  # [N, D]
        feat1_batch = F.normalize(feat1_batch[:, 1:], dim=2)  # [B, N, D]
        sim = torch.einsum('nd,bmd->bnm', feat0, feat1_batch)  # [B, N, N]
        max_sim, _ = sim.max(dim=2)  # [B, N]

        matched = (max_sim > threshold).float()  # [B, N]
        mask = matched.mean(dim=-1) > 0.3

        if return_item:
            return mask, matched.mean(dim=-1)
        else:
            return mask  # [B]

    
    def compute_distance(self, ii, jj):
        N = ii.shape[0]
        distance = []
        for k in range(N):
            if ii[k] == self.keyframes.counter.value - 1 or jj[k] == self.keyframes.counter.value - 1:
                d = np.inf
            else:
                K = self.keyframes.intrinsic[ii[k]]
                pointmap_i = self.keyframes.pointmap[ii[k]]
                pointmap_j = self.keyframes.pointmap[jj[k]]
                # T_i = self.keyframes.pose[ii[k]]
                T_j = self.keyframes.pose[jj[k]]
                quat = T_j[3:]
                trans = T_j[:3]
                rot = R.from_quat(quat.cpu().numpy()).as_matrix()
                c2w = torch.eye(4, device=self.device)
                c2w[:3, :3] = torch.from_numpy(rot)
                c2w[:3, 3] = trans
                T_world_to_cam_j = torch.inverse(c2w)  # [4, 4]

                # Transform pointmap_i to the coordinate frame of T_j
                H, W, _ = pointmap_i.shape
                points_i = pointmap_i.reshape(-1, 3)  # [N, 3]
                ones = torch.ones((points_i.shape[0], 1), device=points_i.device)
                points_i_h = torch.cat([points_i, ones], dim=-1)  # [N, 4]

                # Transform points from world coordinates to the camera frame of T_j
                points_cam_j = (T_world_to_cam_j @ points_i_h.T).T[:, :3]  # [N, 3]

                # Project points to the image plane of frame j
                x = points_cam_j[:, 0]
                y = points_cam_j[:, 1]
                z = points_cam_j[:, 2].clamp(min=1e-5 ,max=10)  ##TODO

                u_proj = (K[0] * x / z + K[2]).round().long()
                v_proj = (K[1] * y / z + K[3]).round().long()

                # Retain valid pixel ranges
                valid = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)
                u_proj = u_proj[valid]
                v_proj = v_proj[valid]

                if valid.sum() < H*W*0.75:
                    d = np.inf
                else:
                    # Get corresponding points from pointmap_i and pointmap_j
                    points_i = points_i[valid]
                    points_j = pointmap_j[v_proj, u_proj]

                    # Compute the reprojection error
                    reprojection_error = torch.norm(points_i - points_j, dim=1)
                    d = reprojection_error.mean().item()
            distance.append(d)
            
        return torch.from_numpy(np.array(distance))

    # def get_covisible_pairs_batch(self, idx_batch):
    #     if not torch.is_tensor(idx_batch):
    #         idx_batch = torch.tensor(idx_batch, device=self.ii.device)

    #     covis_pairs = set()

    #     for idx in idx_batch:
    #         mask = (self.ii == idx) | (self.jj == idx)

    #         ii_rel = self.ii[mask]
    #         jj_rel = self.jj[mask]

    #         for i, j in zip(ii_rel.tolist(), jj_rel.tolist()):
    #             if (i in idx_batch.tolist()):
    #                 pair = tuple((i, j))  
    #                 covis_pairs.add(pair)

    #     return list(covis_pairs)
    
    def get_covisible_pairs_batch(self, idx_batch):
        if not torch.is_tensor(idx_batch):
            idx_batch = torch.tensor(idx_batch, device=self.ii.device)

        ii_list = []
        jj_list = []

        for idx in idx_batch:
            mask = (self.ii == idx) #| (self.jj == idx)

            jj_matches = torch.where(self.ii[mask] == idx, self.jj[mask], self.ii[mask])

            ii_list.append(torch.full_like(jj_matches, idx))
            jj_list.append(jj_matches)

        ii_all = torch.cat(ii_list, dim=0)
        jj_all = torch.cat(jj_list, dim=0)

        return ii_all, jj_all


    def get_covisible_index_batch(self, idx_batch):
        if not torch.is_tensor(idx_batch):
            idx_batch = torch.tensor(idx_batch, device=self.ii.device)

        connected_frames = set()

        # Active factors
        for idx in idx_batch:
            mask = (self.ii == idx) | (self.jj == idx)
            connected_frames.update(self.ii[mask].tolist())
            connected_frames.update(self.jj[mask].tolist())

        # Remove all input idx
        connected_frames.difference_update(idx_batch.tolist())

        return list(connected_frames)

    def get_covisible_index(self, idx):
        """ Find frames connected to the given frame index in the factor graph """
        connected_frames = set()

        # Check active factors
        mask = (self.ii == idx) | (self.jj == idx)
        connected_frames.update(self.ii[mask].tolist())
        connected_frames.update(self.jj[mask].tolist())

        # Remove the input frame index itself
        connected_frames.discard(idx)

        return list(connected_frames)
    
    def get_global_index(self, idx, start_idx=0, num=10, method='sample'):
        if idx < num:
            return list(range(idx))
        elif method == 'sample':
            return [int(i) for i in torch.linspace(start_idx, idx - 1, steps=num)]
        elif method == 'nearest':
            return list(range(idx-num, idx))
        else:
            # Build a graph from the covisible frames
            G = nx.Graph()
            edges = zip(self.ii.cpu().numpy(), self.jj.cpu().numpy())
            G.add_edges_from(edges)

            # Find connected components (clusters)
            clusters = list(nx.connected_components(G))

            representative_frames = []
            cluster_set = clusters[0]
            if len(cluster_set) > num:
                for i in range(num):
                    # Calculate node degrees for each frame in the cluster
                    degrees = {node: G.degree(node) for node in cluster_set}
                    # Select the frame with the highest degree as the representative
                    representative_frame = int(max(degrees, key=degrees.get))
                    representative_frames.append(representative_frame)
                    # Remove the selected frame from the cluster set
                    cluster_set.remove(representative_frame)
            else:
                representative_frames = list(cluster_set)

            return representative_frames
        
    def detect_loop(self, current_idx, current_featI, all_featI,
                temporal_window=8, feat_th=0.7, small_loop_candidates=False):
        """
        Detect loop closure for the current keyframe.

        Prioritize small loop closure using existing covisibility connections,
        then fall back to global loop detection based on image feature similarity.

        Args:
            current_idx (int): Index of the current keyframe.
            current_pts (Tensor): 3D points of the current frame (N, 3).
            current_featI (Tensor): Image feature of the current frame (C, H, W).
            all_c2w (Tensor): Camera poses of all frames (B, 4, 4).
            all_featI (Tensor): Image features of all frames (B, C, H, W).
            K (Tensor): Intrinsic matrix (3, 3).
            temporal_window (int): Minimum frame distance to consider a loop.
            feat_th (float): Feature similarity threshold for global loop detection.

        Returns:
            loop_idx (int or None): The oldest loop frame index if detected, otherwise None.
        """
        device = self.device
        B = all_featI.shape[0]

        with torch.no_grad():
            # Step 1: Small loop detection using existing covisibility graph
            # Get indices of frames that are covisible with the current frame
            covisible_idx = set(self.jj[self.ii == current_idx].tolist())

            # Filter covisible frames that are temporally far enough
            loop_candidates = [
                i for i in covisible_idx if abs(i - current_idx) > temporal_window
            ]

            if loop_candidates:
                if small_loop_candidates:
                    return min(loop_candidates)
                else:
                    return np.array(loop_candidates)
            else:
                return None

            # # Step 2: Global loop detection based on feature similarity
            # all_idx = torch.arange(B, device=device)

            # # Compute feature overlap between current frame and all frames
            # feat_mask = self.compute_feature_overlap_batch(current_featI, all_featI, threshold=feat_th)

            # # Filter frames that are temporally far enough
            # global_loop_candidates = all_idx[feat_mask & (torch.abs(all_idx - current_idx) > temporal_window)]

            # if global_loop_candidates.numel() > 0:
            #     # Return the oldest (smallest index) frame as the global loop closure
            #     loop_idx = torch.min(global_loop_candidates).item()
            #     return loop_idx

        return None
    
    def NMS(self, pointmaps_matched, featI_matched, pose_matched,
            pointmap_current, featI_current, pose_current, K, th=0.3):
        pointmaps_matched = pointmaps_matched.cuda()
        pose_matched = pose_matched.cuda()
        pointmap_current = pointmap_current.cuda()
        pose_current = pose_current.cuda()

        overlap_ratios_a2c = self.cal_overlap_bi(pointmaps_matched, pose_current.unsqueeze(0), K).squeeze()  # [B, 1]
        overlap_ratios_c2a = self.cal_overlap_bi(pointmap_current.unsqueeze(0), pose_matched, K).squeeze()  # [1, B]
        overlap = (overlap_ratios_a2c + overlap_ratios_c2a) / 2

        _, feat_sim = self.compute_feature_overlap_batch(featI_current, featI_matched, return_item=True)

        z_current = pose_current[:3, 2]  # [3]
        z_matched = pose_matched[:, :3, 2]  # [B, 3]
        cos_angles = F.cosine_similarity(z_matched, z_current.unsqueeze(0), dim=1)  # [B]

        scores = 0.8 * overlap + 0.2 * feat_sim #+ 0.1 * torch.clamp(cos_angles, min=0.0)  # [B]
        if scores.max() > th:
            return torch.argmax(scores.cpu())
        else:
            return None


    def vis_graph(self, t1, save_path='covisibility_graph'):
        poses = self.keyframes.pose[:t1]  # (N, 7) --> [q0, q1, q2, q3, x, y, z]

        translations = poses[:, :3].cpu().numpy()  # (N, 3)
        quats = poses[:, 3:].cpu().numpy()          # (N, 4)

        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(translations[:, 0], translations[:, 1], translations[:, 2], c='blue', s=10, label='Keyframes')

        for idx, (x, y, z) in enumerate(translations):
            ax.text(x, y, z, str(idx), color='red', fontsize=8)

        for i, j in zip(ii, jj):
            if i >= t1 or j >= t1:
                continue
            p1 = translations[i]
            p2 = translations[j]
            xs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            zs = [p1[2], p2[2]]
            ax.plot(xs, ys, zs, 'k-', linewidth=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Covisibility Graph at t={t1}')
        ax.legend()

        ax.view_init(elev=0, azim=80)

        pca = PCA(n_components=3)
        pca.fit(translations)

        normal_vector = pca.components_[2]

        # elevation = 90 - angle with Z
        z_axis = np.array([0, 0, 1])
        cos_theta = np.clip(normal_vector @ z_axis, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        elev = np.degrees(angle) - 90

        azim = np.degrees(np.arctan2(normal_vector[1], normal_vector[0]))

        ax.view_init(elev=elev, azim=azim)

        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/covisibility_graph3d_{t1}.png', dpi=300)
        plt.close()


    def visualize_edges(self, N, save_path='covisibility_graph', selected_nodes=None):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        y1 = 1.0
        y2 = 0.0

        if selected_nodes is not None:
            mask = [(i == selected_nodes) for i, j in zip(ii, jj)]
            ii = ii[mask]
            jj = jj[mask]

        segments = [ [(i, y1), (j, y2)] for i, j in zip(ii, jj) ]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_xlim(-1, N)
        ax.set_ylim(-1, 2)

        ax.plot(range(N), [y1]*N, 'o-', color='lightgray', markersize=3)
        ax.plot(range(N), [y2]*N, 'o-', color='lightgray', markersize=3)

        lc = LineCollection(segments, colors='blue', linewidths=0.8, alpha=0.9)
        ax.add_collection(lc)

        ax.set_yticks([y1, y2])
        ax.set_yticklabels(['ii', 'jj'])
        ax.set_xlabel("Keyframe Index")
        ax.set_title("Covisibility Graph (Selected)")
        ax.legend()

        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/covisibility_graph_{selected_nodes}.png', dpi=300)
        plt.close()