import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from lietorch import SE3
from src.dust3r.inference import inference
from src.dust3r.utils.camera import pose_encoding_to_camera
from src.dust3r.utils.geometry import geotrf
from factor_graph import FactorGraph
from scipy.spatial.transform import Rotation
from util.utils import viz_pcd, pose_vec_to_matrix, umeyama_alignment


class TrackBackend:
    def __init__(self, slam, keyframes, config, device="cuda:0"):
        self.device = device
        self.keyframes = keyframes
        self.model = slam.model
        self.graph = FactorGraph(self.keyframes, max_factors=48)
        self.graph = slam.graph

        # local optimization window
        self.t1 = 0

        # frontent variables
        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2
        self.warmup = 6

        self.frontend_nms = config["frontend_nms"]
        self.keyframe_thresh = config["keyframe_thresh"]
        self.frontend_window = config["frontend_window"]
        self.frontend_thresh = config["frontend_thresh"]
        self.frontend_radius = config["frontend_radius"]
        self.loop_iters = config["iteration"]
        self.keyframes.mono_depth_alpha = config["mono_depth_alpha"]
        self.ba_window = 5

        self.verbose = slam.verbose
        self.output_dir = slam.output_dir

        self.conf_th = 0.05
        self.downsample_ratio = slam.downsample_ratio

        # save closed loop
        self.lc_initialized = False
        self.closed_loop = {}
        self.closed_loop['idx_current'] = []
        self.closed_loop['idx_matched'] = []
        self.closed_loop['pointmaps_lc'] = []

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
    
    def prepare_input(self, images):
        images = self.model.normalize(images).to('cuda')
        views = []
        for i in range(len(images)):
            view = {
                "img": images[i][None],
                "ray_map": torch.full(
                    (
                        1,
                        6,
                        images[i].shape[-2],
                        images[i].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(np.int32([images[i].shape[-2], images[i].shape[-1]])),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)

        return views

    def prepare_output(self, outputs, use_pose=True):
        outputs["pred"] = outputs["pred"]
        outputs["views"] = outputs["views"]

        pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
        pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
        conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
        conf_other = [output["conf"].cpu() for output in outputs["pred"]]
        pts3ds_self = torch.cat(pts3ds_self_ls, 0)

        # Recover camera poses.
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in outputs["pred"]
        ]  # [N, 4, 4]

        if use_pose:
            transformed_pts3ds_other = []
            for pose, pself in zip(pr_poses, pts3ds_self):
                transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
            pts3ds_other = torch.cat(transformed_pts3ds_other, 0)
            conf_other = torch.cat(conf_self, 0)

        return pts3ds_self, pts3ds_other, conf_other, torch.cat(pr_poses, 0)
    
    def track(self, selected_idx, anchor_sub_num):
       
        pts_all = []
        confs_all = []
        poses_all = []
        img_all = []
        
        selected_idx = selected_idx.numpy()
        B = len(selected_idx)

        imgs = self.keyframes.image[selected_idx]

        views = self.prepare_input(imgs)
        outputs, _ = inference(views, self.model, self.device)
        pts3ds_self, pts3ds_other, confs, poses = self.prepare_output(outputs)
        depths = pts3ds_self[..., 2]

        first_c2w = poses[0]
        first_w2c = torch.inverse(first_c2w)

        for i in range(B):
            intrinsic = self.keyframes.intrinsic[i].cpu().data.numpy()
            K = np.array([[intrinsic[0], 0, intrinsic[2]],[0, intrinsic[1], intrinsic[3]],[0,0,1]])

            img = imgs[i]
            pts = pts3ds_self[i]
            conf = confs[i]
            conf = 1 - 1/ conf
            # depth = depths[i]
            pose = poses[i]

            if i == 0:
                # align with previous submap
                umeyama = False
                if umeyama:
                    prev_pts = self.keyframes.submap_ds[anchor_sub_num][0].detach().cpu().numpy().reshape(-1, 3)  # [H*W, 3]
                    pts_ds = pts[::self.downsample_ratio, ::self.downsample_ratio].cpu().numpy().reshape(-1, 3)
                    conf_np = conf.cpu().numpy()
                    conf_mask = (conf_np[::self.downsample_ratio, ::self.downsample_ratio] > self.conf_th).reshape(-1)  # [H, W]
                    align_s, align_R, align_t = umeyama_alignment(pts_ds[conf_mask], prev_pts[conf_mask])
                    align_R = torch.from_numpy(align_R).float()
                    align_t = torch.from_numpy(align_t).float()
                else:
                    prev_depth = self.keyframes.depth[anchor_sub_num * 5].cpu()
                    depth = depths[i].cpu()
                    log_scale = (torch.log(prev_depth) - torch.log(depth)).mean()
                    align_s = torch.exp(log_scale)
                    prev_c2w = pose_vec_to_matrix(self.keyframes.pose[anchor_sub_num * 5].unsqueeze(0).cpu())[0]
                    align_R = prev_c2w[:3, :3]
                    align_t = prev_c2w[:3, 3]

            pose = first_w2c @ pose
            R = pose[:3, :3]
            T = pose[:3, 3]
            R_aligned = torch.matmul(align_R, R)
            T_aligned = torch.matmul(align_R, align_s * T) + align_t
            pose_aligned = torch.eye(4)
            pose_aligned[:3, :3] = R_aligned
            pose_aligned[:3, 3] = T_aligned
            pointmap = geotrf(pose_aligned.unsqueeze(0), align_s * pts.unsqueeze(0))[0]
            pose = pose_aligned
            r = Rotation.from_matrix(pose[:3, :3].cpu().numpy())
            quat = torch.from_numpy(r.as_quat())
            trans = pose[:3, 3]
              
            pts_all.append(pointmap.to(self.device))  # [H, W, 3]
            confs_all.append(conf)  # rearrange to [0, 1]
            poses_all.append(torch.cat([trans, quat], dim=-1))  # c2w
            img_all.append(img.squeeze(0))
            
        pointmaps = torch.stack(pts_all, dim=0)[:, ::self.downsample_ratio, ::self.downsample_ratio]
        confs = torch.stack(confs_all, dim=0)[:, ::self.downsample_ratio, ::self.downsample_ratio]
        images = torch.stack(img_all, dim=0)[:, :, ::self.downsample_ratio, ::self.downsample_ratio]
        poses = torch.stack(poses_all, dim=0)

        if self.verbose:
            pts_np = pointmaps.detach().cpu().reshape(-1, 3).numpy()
            img_np = images.permute(0, 2, 3, 1).detach().cpu().reshape(-1, 3).numpy() / 255.0
            viz_pcd(pts_np, img_np, os.path.join(self.output_dir, "backend"), name=f"frame_{anchor_sub_num*5}.ply")

        return pointmaps, confs, poses
    

    def loop_closure_init(self, pointmap_current_lc, idx_matched, idx_current):
        with torch.enable_grad():
            sub_num_matched = 0 #idx_matched // 5
            sub_num_current = idx_current // 5

            # prepare pointmaps for lc
            ## from matched submap to current submap
            pointmaps_all_batch = self.keyframes.submap_ds[sub_num_matched:sub_num_current + 1]  # [B, N, H, W, 3]
            pointmaps_all_batch = pointmaps_all_batch.to(self.device)
            confs_all_batch = self.keyframes.conf_ds[sub_num_matched:sub_num_current + 1]  # [B, N, H, W]
            confs_all_batch = confs_all_batch.to(self.device)
            confs = confs_all_batch[:-1, -1, ...]
            confs_mask = confs > self.conf_th * 0  ##TODO: check the th vlaue

            ## the current frame's pointmap (global coordinate)
            pointmap_current = self.keyframes.submap_ds[sub_num_current, idx_current%5]  # [H, W, 3]
            pointmap_current = pointmap_current.to(self.device)
            conf_current = self.keyframes.conf_ds[sub_num_current, idx_current%5]  # [H, W, 3]
            conf_current = conf_current.to(self.device)
            conf_mask_current = conf_current > self.conf_th * 0  ##TODO: check the th vlaue

            ## current pointmap (matched batch coordinate)
            pointmap_current_lc = pointmap_current_lc.reshape(1, -1, 3)

            # flatten
            B, N, H, W, _ = pointmaps_all_batch.shape
            confs_mask = confs_mask.reshape(B - 1, -1)
            conf_mask_current = conf_mask_current.reshape(1, -1)

            if self.verbose:  
                pts = (self.keyframes.submap_ds[:sub_num_current + 1, :-1].flatten(0, 1))[:idx_current // 5 * 5 + 5].detach().cpu().numpy()
                confs = (self.keyframes.conf_ds[:sub_num_current + 1, :-1].flatten(0, 1))[:idx_current // 5 * 5 + 5].detach().cpu().numpy()
                images = self.keyframes.image[:idx_current // 5 * 5 + 5].permute(0, 2, 3, 1).detach().cpu().numpy() / 255.0
                images = images[:, ::self.downsample_ratio, ::self.downsample_ratio]
                viz_pcd(pts, images, pcd_dir=os.path.join(self.output_dir, "LC"), name=f"frame_{idx_current}_before.ply", conf=confs, th=self.conf_th)
         
            # Set optimizer
            _align_lie = torch.nn.Parameter(torch.zeros(B - 1, 6, device=self.device)) 
            lie_0 = torch.zeros(1, 6, device=self.device, dtype=torch.float)
            optimizer = torch.optim.Adam([{'params': _align_lie, 'lr': 0.0005}])
            
            pointmaps_first_last = torch.stack([pointmaps_all_batch[:, 0], pointmaps_all_batch[:, -1]], dim=1)  #[B, 2, H, W, 3]
            
            pbar = tqdm(range(self.loop_iters), desc="Loop Closure")
            for iter in range(self.loop_iters):
                optimizer.zero_grad()

                align_lie = torch.cat([lie_0, _align_lie], dim=0)

                se3_new = SE3.exp(align_lie)  # [B]
                T_s2t = se3_new.matrix()   

                # T_s2t = pose_vec_to_matrix(align_quat)  # [B, 4, 4]
                align_R = T_s2t[:, :3, :3]
                align_t = T_s2t[:, :3, 3].unsqueeze(1)

                # align the last batch with first batch: pointmaps after lc <-- current pointmap
                pts_current = pointmap_current.view(1, -1, 3)
                pts_current_aligned = torch.matmul(pts_current, align_R[-1].unsqueeze(0).transpose(1, 2)) + align_t[-1].unsqueeze(0)
                current_lc_loss = torch.abs((pts_current_aligned - pointmap_current_lc)).mean()

                # align current first pointmap with previous last pointmap: last_i-1 <-- first_i
                pts_fl = pointmaps_first_last.view(B, 2, -1, 3)
                pts_fl_aligned = torch.matmul(pts_fl, align_R.transpose(1, 2).unsqueeze(1)) + align_t.unsqueeze(1)

                pts_prev_last = pts_fl_aligned[:-1, -1, ...]
                pts_next_first = pts_fl_aligned[1:, 0, ...]
                fl_loss = torch.abs((pts_prev_last - pts_next_first)[confs_mask]).mean()

                loss = fl_loss + current_lc_loss
                
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update()

        align_lie = torch.cat([lie_0, _align_lie], dim=0)
        se3_new = SE3.exp(align_lie)  # [B-1]
        T_s2t = se3_new.matrix() 
        align_R = T_s2t[:, :3, :3]
        align_t = T_s2t[:, :3, 3].unsqueeze(1)
        
        # update all pointmaps (fix the first submap)
        _pts = pointmaps_all_batch  # -> [B, N, H, W, 3]
        _rotated = torch.matmul(_pts, align_R.transpose(1, 2).unsqueeze(1).unsqueeze(1))  # [B, N, H, W, 3] x [B, 1, 1, 3, 3] -> [B, N, H, W, 3]
        pointmap_aligned = _rotated + align_t.view(B, 1, 1, 1, 3)  # -> [B, N, H, W, 3]
        self.keyframes.submap_ds[sub_num_matched:sub_num_current + 1] = pointmap_aligned.cpu()

        # update pose
        orig_pose = self.keyframes.pose[sub_num_matched * 5 : (sub_num_current + 1) * 5].to(self.device)  # [B*N, 7]
        orig_pose = orig_pose.view(B, -1, 7)
        new_pose = []
        for b in range(B):
            for n in range(N - 1):
                orig_c2w = pose_vec_to_matrix(orig_pose[b, n].unsqueeze(0))[0]  # [4,4]
                
                T_sim = torch.eye(4, device=orig_c2w.device)
                T_sim[:3, :3] = align_R[b]
                T_sim[:3, 3] = align_t[b, 0]

                T_new = T_sim @ orig_c2w
                
                R_new = T_new[:3, :3].cpu().numpy()
                t_new = T_new[:3, 3].cpu().numpy()
                quat_new = Rotation.from_matrix(R_new).as_quat()
                pose7d = torch.from_numpy(np.concatenate([t_new, quat_new])).float()
                new_pose.append(pose7d)

        new_pose = torch.stack(new_pose, dim=0).reshape(-1, 7)
        self.keyframes.pose[sub_num_matched * 5 : (sub_num_current + 1) * 5] = new_pose.cpu()

        # update the last pose
        last_orig_pose = self.keyframes.pose[(sub_num_current + 1) * 5].to(self.device)
        last_orig_c2w = pose_vec_to_matrix(last_orig_pose.unsqueeze(0))[0]  # [4,4]
        T_sim = torch.eye(4, device=last_orig_c2w.device)
        T_sim[:3, :3] = align_R[-1]
        T_sim[:3, 3] = align_t[-1, 0]
        T_new = T_sim @ last_orig_c2w
        R_new = T_new[:3, :3].cpu().numpy()
        t_new = T_new[:3, 3].cpu().numpy()
        quat_new = Rotation.from_matrix(R_new).as_quat()
        last_pose7d = torch.from_numpy(np.concatenate([t_new, quat_new])).float()
        self.keyframes.pose[(sub_num_current + 1) * 5] = last_pose7d.cpu()

        if self.verbose:
            pts = (self.keyframes.submap_ds[:sub_num_current + 1, :-1].flatten(0, 1))[:idx_current // 5 * 5 + 5].detach().cpu().numpy()
            confs = (self.keyframes.conf_ds[:sub_num_current + 1, :-1].flatten(0, 1))[:idx_current // 5 * 5 + 5].detach().cpu().numpy()
            images = self.keyframes.image[:idx_current // 5 * 5 + 5].permute(0, 2, 3, 1).detach().cpu().numpy() / 255.0
            images = images[:, ::self.downsample_ratio, ::self.downsample_ratio]
            viz_pcd(pts, images, pcd_dir=os.path.join(self.output_dir, "LC"), name=f"frame_{idx_current}_after.ply", conf=confs, th=self.conf_th)

        updates = {
            'pose_updates': se3_new.data,
            'submap_idx': range(sub_num_matched, sub_num_current + 1),
            'camera_idx': range(sub_num_matched * 5, (sub_num_current + 1) * 5 + 1),
            'camera_pose': torch.cat([new_pose.reshape(-1, 7).cpu(), last_pose7d.unsqueeze(0).cpu()], dim=0)
        }
        return updates
    

    def loop_closure(self, pointmaps_lc, idx_matched, idx_current):
        sub_num_oldest = 0
        sub_num_current = idx_current // 5

        # prepare all pointmap batches
        pointmaps_all_batch = self.keyframes.submap_ds[sub_num_oldest:sub_num_current + 1].cuda()  # [B, N, H, W, 3]
        pointmaps_first_last = torch.stack([pointmaps_all_batch[:, 0], pointmaps_all_batch[:, -1]], dim=1)  #[B, 2, H, W, 3]
        pointmaps_first_last = pointmaps_first_last.to(self.device)

        # prepare current pointmaps (previous current + new current) (global coordinate)
        B, N, H, W, _ = pointmaps_all_batch.shape
        idx_current_prevs = np.array(self.closed_loop['idx_current'])
        sub_num_current_prevs = idx_current_prevs // 5  # previsou loop closure submap idx
        sub_num_current_all = np.append(sub_num_current_prevs, sub_num_current)  # all loop closure submap idx

        pointmap_current = self.keyframes.submap_ds[sub_num_current, idx_current%5].cuda()  # [H, W, 3]
        pointmap_current_prevs = self.keyframes.submap_ds[sub_num_current_prevs, idx_current_prevs%5].cuda()  # [H, W, 3]
        pointmap_current = torch.cat([pointmap_current_prevs, pointmap_current.unsqueeze(0)], dim=0)
        pointmap_current = pointmap_current.to(self.device)
        
        # current pointmaps (matched batch coordinate)
        pointmaps_lc_prevs = torch.stack(self.closed_loop['pointmaps_lc'], dim=0)
        pointmaps_lc = torch.cat([pointmaps_lc_prevs, pointmaps_lc.unsqueeze(0)], dim=0).float()
        pointmaps_lc_first_last = torch.stack([pointmaps_lc[:, 0], pointmaps_lc[:, -1]], dim=1).float()

        Bc, Nc, H, W, _ = pointmaps_lc.shape

        idx_matched_prevs = np.array(self.closed_loop['idx_matched'])
        sub_num_matched_prevs = idx_matched_prevs // 5
        sub_num_matched_all = np.append(sub_num_matched_prevs, idx_matched // 5)

        if self.verbose:  
            pts = (self.keyframes.submap_ds[:sub_num_current + 1, :-1].flatten(0, 1))[:idx_current//5*5+5].detach().cpu().numpy()
            confs = (self.keyframes.conf_ds[:sub_num_current + 1, :-1].flatten(0, 1))[:idx_current//5*5+5].detach().cpu().numpy()
            images = self.keyframes.image[:idx_current//5*5+5].permute(0, 2, 3, 1).detach().cpu().numpy() / 255.0
            images = images[:, ::self.downsample_ratio, ::self.downsample_ratio]
            viz_pcd(pts, images, pcd_dir=os.path.join(self.output_dir, "LC"), name=f"frame_{idx_current}_before.ply", conf=confs, th=self.conf_th)
    
        # pose graph optimization
        with torch.enable_grad():
            _align_lie = torch.nn.Parameter(torch.zeros(B - 1, 6, device=self.device)) 

            # align loop closure pointmaps with matched pointmap
            matched_lie = torch.nn.Parameter(torch.zeros(Bc, 6, device=self.device)) 
            
            optimizer = torch.optim.Adam([{'params': _align_lie, 'lr': 0.0005},
                                          {'params': matched_lie, 'lr': 0.0005}
                                        ])
            # fix the first submap
            lie_0 = torch.zeros(1, 6, device=self.device, dtype=torch.float)

            pbar = tqdm(range(self.loop_iters), desc="Loop Closure")
            for iter in range(self.loop_iters):
                optimizer.zero_grad()

                align_lie = torch.cat([lie_0, _align_lie], dim=0)

                se3_align = SE3.exp(align_lie)  # [B-1]
                T_s2t = se3_align.matrix()   
                # T_s2t = pose_vec_to_matrix(align_quat)  # [B, 4, 4]
                align_R = T_s2t[:, :3, :3]
                align_t = T_s2t[:, :3, 3].unsqueeze(1)

                se3_matched = SE3.exp(matched_lie)  # [B-1]
                matched_T_s2t = se3_matched.matrix()   
                # matched_T_s2t = pose_vec_to_matrix(matched_quat)
                matched_R = matched_T_s2t[:, :3, :3]
                matched_t = matched_T_s2t[:, :3, 3].unsqueeze(1)

                # transform submaps
                pts_fl = pointmaps_first_last.view(B, 2, -1, 3)
                pts_fl_aligned = torch.matmul(pts_fl, align_R.transpose(1, 2).unsqueeze(1)) + align_t.unsqueeze(1)

                # transform lc submaps
                pts_lc_fl = pointmaps_lc_first_last.view(Bc, 2, -1, 3)
                pointmaps_lc_fl_aligned = torch.matmul(pts_lc_fl, matched_R.transpose(1, 2).unsqueeze(1)) + matched_t.unsqueeze(1)
                
                # transform current pointmap
                pts_current = pointmap_current.view(Bc, -1, 3)
                pts_current_aligned = torch.matmul(pts_current, align_R[sub_num_current_all].transpose(1, 2)) + align_t[sub_num_current_all]
                
                # align the first pointmap of current submap with the last pointmap of previous submap
                fl_loss = torch.abs((pts_fl_aligned[:-1, -1, ...] - pts_fl_aligned[1:, 0, ...])).mean()
                # align the lc submaps with corresponding matched submaps(transformed)
                matched_loss = torch.abs(pointmaps_lc_fl_aligned[:, 0] - pts_fl_aligned[sub_num_matched_all, 0]).mean()
                # align the current pointmap with corresponding pointmap in lc submaps
                current_lc_loss = torch.abs((pts_current_aligned - pointmaps_lc_fl_aligned[:, -1])).mean()
                # total loss
                loss = fl_loss + current_lc_loss + matched_loss

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update()

        align_lie = torch.cat([lie_0, _align_lie], dim=0)
        se3_align = SE3.exp(align_lie)
        T_s2t = se3_align.matrix()   
        align_R = T_s2t[:, :3, :3]
        align_t = T_s2t[:, :3, 3].unsqueeze(1)

        # update pointmaps
        _pts = pointmaps_all_batch  # -> [B, N, H, W, 3]
        _rotated = torch.matmul(_pts, align_R.transpose(1, 2).unsqueeze(1).unsqueeze(1))
        pointmap_aligned = _rotated + align_t.view(B, 1, 1, 1, 3)  # -> [B, N, H, W, 3]
        self.keyframes.submap_ds[sub_num_oldest:sub_num_current + 1] = pointmap_aligned.cpu()

        # update pose
        orig_pose = self.keyframes.pose[sub_num_oldest * 5 : (sub_num_current + 1) * 5].to(self.device)  # [B*N, 7]
        orig_pose = orig_pose.view(B, N - 1, 7)
        new_pose = []
        for b in range(B):
            for n in range(N - 1):
                orig_c2w = pose_vec_to_matrix(orig_pose[b, n].unsqueeze(0))[0]  # [4,4]

                T_sim = torch.eye(4, device=orig_c2w.device)
                T_sim[:3, :3] = align_R[b]
                T_sim[:3, 3] = align_t[b, 0]

                T_new = T_sim @ orig_c2w
                
                R_new = T_new[:3, :3].cpu().numpy()
                t_new = T_new[:3, 3].cpu().numpy()
                quat_new = Rotation.from_matrix(R_new).as_quat()
                pose7d = torch.from_numpy(np.concatenate([t_new, quat_new])).float()
                new_pose.append(pose7d)

        new_pose = torch.stack(new_pose, dim=0).reshape(-1, 7)
        self.keyframes.pose[sub_num_oldest * 5 : (sub_num_current + 1) * 5] = new_pose.cpu()

        # update the last pose
        orig_pose = self.keyframes.pose[(sub_num_current + 1) * 5].to(self.device)
        orig_c2w = pose_vec_to_matrix(orig_pose.unsqueeze(0))[0]  # [4,4]
        T_sim = torch.eye(4, device=orig_c2w.device)
        T_sim[:3, :3] = align_R[-1]
        T_sim[:3, 3] = align_t[-1, 0]
        T_new = T_sim @ orig_c2w
        R_new = T_new[:3, :3].cpu().numpy()
        t_new = T_new[:3, 3].cpu().numpy()
        quat_new = Rotation.from_matrix(R_new).as_quat()
        last_pose7d = torch.from_numpy(np.concatenate([t_new, quat_new])).float()
        self.keyframes.pose[(sub_num_current + 1) * 5] = last_pose7d.cpu()

        # update loop closure pointmaps
        pointmaps_lc_aligned = torch.matmul(pointmaps_lc.view(Bc, Nc, -1, 3), matched_R.transpose(1, 2).unsqueeze(1)) + matched_t.unsqueeze(1)
        pointmaps_lc_aligned = pointmaps_lc_aligned.reshape(Bc, -1, H, W, 3)
        for i in range(Bc - 1):
            self.closed_loop['pointmaps_lc'][i] = pointmaps_lc_aligned[i]

        if self.verbose:
            pts = (self.keyframes.submap_ds[:sub_num_current + 1, :-1].flatten(0, 1))[:idx_current+1].detach().cpu().numpy()
            confs = (self.keyframes.conf_ds[:sub_num_current + 1, :-1].flatten(0, 1))[:idx_current+1].detach().cpu().numpy()
            images = self.keyframes.image[:idx_current+1].permute(0, 2, 3, 1).detach().cpu().numpy() / 255.0
            images = images[:, ::self.downsample_ratio, ::self.downsample_ratio]
            viz_pcd(pts, images, pcd_dir=os.path.join(self.output_dir, "LC"), name=f"frame_{idx_current}_after.ply", conf=confs, th=self.conf_th)
        
        updates = {
            'pose_updates': se3_align.data,
            'submap_idx': range(0, sub_num_current + 1),
            'camera_idx': range(0, (sub_num_current + 1) * 5 + 1),
            'camera_pose': torch.cat([new_pose.reshape(-1, 7).cpu(), last_pose7d.unsqueeze(0).cpu()], dim=0)
        }
        return pointmaps_lc_aligned[-1], updates
    

    def run(self, ):
        intrinsic = self.keyframes.intrinsic[0].cpu().data.numpy() / self.downsample_ratio
        K = np.array([[intrinsic[0], 0, intrinsic[2]],[0, intrinsic[1], intrinsic[3]],[0,0,1]])

        # Loop detection
        t1 = self.keyframes.counter.value - 1
        t0 = t1 - 6
        for idx_current in range(t0, t1 - 1):
            featI_current = self.keyframes.featI[idx_current]
            featI_all = self.keyframes.featI[:idx_current]
            
            ids_matched = self.graph.detect_loop(idx_current, featI_current, featI_all, small_loop_candidates=False)
            if ids_matched is not None:
                break
        
        if ids_matched is None:
            return False, None  # no matched loop closure frame, quit

        # NMS: find the most fitted frame
        pointmaps_matched = self.keyframes.submap_ds[ids_matched // 5, ids_matched % 5]
        featI_matched = self.keyframes.featI[ids_matched]
        pose_matched = self.keyframes.pose[ids_matched]
        c2w_matched = pose_vec_to_matrix(pose_matched)
        pointmap_current = self.keyframes.submap_ds[idx_current // 5, idx_current % 5]
        pose_current = self.keyframes.pose[idx_current]
        c2w_current = pose_vec_to_matrix(pose_current[None])[0]

        k_th = self.graph.NMS(pointmaps_matched, featI_matched, c2w_matched, pointmap_current, featI_current, c2w_current, K)

        if k_th is None:
            return False, None

        idx_matched = ids_matched[k_th]
        print(f'Loop Closured: idx matched: {idx_matched}, idx current: {idx_current}')

        # track with matched keyframes
        anchor_sub_num = idx_matched // 5
        ids_previous = torch.arange(anchor_sub_num * 5, (anchor_sub_num + 1) * 5)  # including last idx
        ids_current = torch.arange(idx_current, idx_current-1, -1)
        selected_idx = torch.cat([ids_previous, ids_current])  # [6, 1]

        # track and align with matched previous keyframes
        pointmaps_lc, confs_lc, poses_lc = self.track(selected_idx, anchor_sub_num)
        pointmap_current_lc = pointmaps_lc[-1].unsqueeze(0).to(self.device)

        # Close the loop: submap level BA
        if not self.lc_initialized:
            updates= self.loop_closure_init(pointmap_current_lc, idx_matched, idx_current)
            self.lc_initialized = True
            # self.graph.update((idx_matched // 5) * 5, (idx_current // 5 + 1) * 5, K)
        else:
            pointmaps_lc, updates = self.loop_closure(pointmaps_lc, idx_matched, idx_current)
            # self.graph.update(0, (idx_current // 5 + 1) * 5, K)

        # save closed loop information
        self.closed_loop['idx_current'].append(idx_current)
        self.closed_loop['idx_matched'].append(idx_matched)
        self.closed_loop['pointmaps_lc'].append(pointmaps_lc)
        
        return True, updates

