import os
import torch
# import lietorch
# import cv2
import numpy as np
# import open3d as o3d
# from tqdm import tqdm, trange
from src.dust3r.inference import inference
from src.dust3r.utils.camera import pose_encoding_to_camera
# from src.dust3r.post_process import estimate_focal_knowing_depth
from src.dust3r.utils.geometry import geotrf
from factor_graph import FactorGraph
from scipy.spatial.transform import Rotation
from util.utils import viz_pcd, pose_vec_to_matrix, viz_map, depth_to_pointmap, umeyama_alignment

class TrackFrontend:
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
        self.keyframes.mono_depth_alpha = config["mono_depth_alpha"]
        self.ba_window = 5

        self.verbose = slam.verbose
        self.output_dir = slam.output_dir
        self.use_gt = slam.use_gt

        self.conf_th = 0.05
        self.downsample_ratio = slam.downsample_ratio

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
    
    def predict(self, new_img, kf_img, kf_pose, kf_depth, kf_pointmap):
        imgs = torch.stack([kf_img, new_img], dim=0)

        views = self.prepare_input(imgs)
        outputs, _ = inference(views, self.model, self.device)
        pts3ds_self, pts3ds_other, confs, poses = self.prepare_output(outputs)
        depths = pts3ds_self[..., 2]

        pts = pts3ds_self[0]
        conf = confs[0]
        depth = depths[0]
        umeyama = False
        if umeyama:
            prev_pts = kf_pointmap.detach().cpu().numpy().reshape(-1, 3)  # [H*W, 3]
            
            pts_ds = pts[::self.downsample_ratio, ::self.downsample_ratio].cpu().numpy().reshape(-1, 3)
            conf_np = conf.cpu().numpy()
            conf_mask = (conf_np[::self.downsample_ratio, ::self.downsample_ratio] > 0.2).reshape(-1)  # [H, W]
            if conf_mask.sum() < 50:
                conf_mask = np.ones_like(conf_mask).astype(bool)
            align_s, align_R, align_t = umeyama_alignment(pts_ds[conf_mask], prev_pts[conf_mask])
            align_R = torch.from_numpy(align_R).float()
            align_t = torch.from_numpy(align_t).float()
        else:
            prev_depth = kf_depth.cpu()
            depth = depth.cpu()
            log_scale = (torch.log(prev_depth) - torch.log(depth)).mean()
            align_s = torch.exp(log_scale)
            prev_c2w = pose_vec_to_matrix(kf_pose.unsqueeze(0))[0]
            align_R = prev_c2w[:3, :3]
            align_t = prev_c2w[:3, 3]

        # align new frame to kf
        first_c2w = poses[0]
        first_w2c = torch.inverse(first_c2w)
        pose = poses[1]
        pose = first_w2c @ pose
        pts = pts3ds_self[1]
        depth = depths[1]

        R = pose[:3, :3]
        T = pose[:3, 3]
        R_aligned = torch.matmul(align_R, R)
        T_aligned = torch.matmul(align_R, align_s * T) + align_t
        pose_aligned = torch.eye(4)
        pose_aligned[:3, :3] = R_aligned
        pose_aligned[:3, 3] = T_aligned
        pointmap = geotrf(pose_aligned.unsqueeze(0), align_s * pts.unsqueeze(0))[0]
        depth = align_s * depth
        pose = pose_aligned

        r = Rotation.from_matrix(pose[:3, :3].cpu().numpy())
        quat = torch.from_numpy(r.as_quat())
        trans = pose[:3, 3]

        new_pointmap = pointmap[::self.downsample_ratio, ::self.downsample_ratio]  # [H//d, W//d, 3]      
        new_conf = conf[::self.downsample_ratio, ::self.downsample_ratio]
        new_pose = torch.cat([trans, quat], dim=-1)  # c2w            
        new_depth = depth.cpu()

        return new_pose, new_depth, new_pointmap, new_conf
            
    

    def track(self, t0, t1, init=False):
        '''
            If init=True, initialize the tracker and factor graph, else track latest keyframes.
        '''
        if init:
            # graph initialization
            self.graph.add_neighborhood_factors(0, 3, r=3)

        imgs = self.keyframes.image[t0:t1]

        views = self.prepare_input(imgs)
        outputs, _ = inference(views, self.model, self.device)
        pts3ds_self, pts3ds_other, confs, poses = self.prepare_output(outputs)
        depths = pts3ds_self[..., 2]

        first_c2w = poses[0]
        first_w2c = torch.inverse(first_c2w)

        # update keyframes and factor graph
        sub_num = t0 // 5
        for i in range(t0, t1):
            if not init:
                self.graph.add_neighborhood_factors(i-3, i+1, r=3)

            img = imgs[i - t0]
            pts = pts3ds_self[i - t0]
            conf = confs[i - t0]
            conf = 1 - 1/ conf
            depth = depths[i - t0]
            pose = poses[i - t0]

            if init:
                pointmap = pts3ds_other[i - t0]
            else:
                if i == t0:
                    # align with previous submap
                    umeyama = False
                    if umeyama:
                        prev_pts = self.keyframes.submap_ds[sub_num-1][-1].detach().cpu().numpy().reshape(-1, 3)  # [H*W, 3]
                        
                        pts_ds = pts[::self.downsample_ratio, ::self.downsample_ratio].cpu().numpy().reshape(-1, 3)
                        conf_np = conf.cpu().numpy()
                        conf_mask = (conf_np[::self.downsample_ratio, ::self.downsample_ratio] > 0.2).reshape(-1)  # [H, W]
                        if conf_mask.sum() < 50:
                            conf_mask = np.ones_like(conf_mask).astype(bool)
                        align_s, align_R, align_t = umeyama_alignment(pts_ds[conf_mask], prev_pts[conf_mask])
                        align_R = torch.from_numpy(align_R).float()
                        align_t = torch.from_numpy(align_t).float()
                    else:
                        prev_depth = self.keyframes.depth[i].cpu()
                        depth = depth.cpu()
                        log_scale = (torch.log(prev_depth) - torch.log(depth)).mean()
                        align_s = torch.exp(log_scale)
                        prev_c2w = pose_vec_to_matrix(self.keyframes.pose[i].unsqueeze(0))[0]
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
                depth = align_s * depth
                pose = pose_aligned

            r = Rotation.from_matrix(pose[:3, :3].cpu().numpy())
            quat = torch.from_numpy(r.as_quat())
            trans = pose[:3, 3]

            self.keyframes.submap_ds[sub_num, i-t0] = pointmap[::self.downsample_ratio, ::self.downsample_ratio]  # [H//d, W//d, 3]      
            self.keyframes.conf_ds[sub_num, i-t0] = conf[::self.downsample_ratio, ::self.downsample_ratio]
            self.keyframes.pose[i] = torch.cat([trans, quat], dim=-1)  # c2w            
            self.keyframes.depth[i] = depth.cpu()
            
            # covisibility graph update
            if i > 2:
                H, W, _ = pointmap.shape
                all_poses = self.keyframes.pose[:i].to(self.device)  # [i, 7]
                all_c2ws = pose_vec_to_matrix(all_poses)
                if sub_num > 0:
                    all_pointmaps = self.keyframes.submap_ds[:sub_num, :-1].reshape(-1, H//self.downsample_ratio, W//self.downsample_ratio, 3)
                    all_pointmaps = torch.cat([all_pointmaps, self.keyframes.submap_ds[sub_num, :i-t0]], dim=0).to(self.device) 
                else:
                    all_pointmaps = self.keyframes.submap_ds[sub_num, :i-t0].to(self.device) 

                current_pose = self.keyframes.pose[i].to(self.device)  # [7]
                current_c2w = pose_vec_to_matrix(current_pose.unsqueeze(0))[0]  # [1, 4, 4]                
                current_pointmap = pointmap.to(self.device) 

                intrinsic = self.keyframes.intrinsic[i].cpu().data.numpy()
                K = np.array([[intrinsic[0], 0, intrinsic[2]],[0, intrinsic[1], intrinsic[3]],[0,0,1]])
                self.graph.add(i, all_c2ws, all_pointmaps, current_c2w, current_pointmap, K)  # compute reprojection error and view direction and distance

            if self.verbose and i > 0:
                depth_dir = f"{self.output_dir}/depth"
                os.makedirs(depth_dir, exist_ok=True)

                # viz_depth(W, H, target_pts, w2c, output_dir)
                output_dir = f"{depth_dir}/depth_{i}_frame_{int(self.keyframes.tstamp[i].cpu().numpy())}.png"
                viz_map(depths[0].detach().cpu().numpy(), output_dir, colorize=True)
                output_dir = f"{depth_dir}/pointmap_{i}_frame_{int(self.keyframes.tstamp[i].cpu().numpy())}.png"
                viz_map(((pointmap - pointmap.min()) / (pointmap.max() - pointmap.min() + 1e-8)).detach().cpu().numpy(), output_dir, colorize=False)
                output_dir = f"{depth_dir}/rgb_{i}_frame_{int(self.keyframes.tstamp[i].cpu().numpy())}.png"
                viz_map(((img.squeeze(0).permute(1, 2, 0)/255.0)).detach().cpu().numpy(), output_dir, colorize=False)

                pts_np = pointmap.detach().cpu().reshape(-1, 3).numpy()
                img_np = img.squeeze(0).permute(1, 2, 0).detach().cpu().reshape(-1, 3).numpy()/255.0
                conf_np = conf.detach().cpu().reshape(-1).numpy()
                viz_pcd(pts_np, img_np, os.path.join(self.output_dir, "pcd_per_frame"), name=f"frame_{i}.ply", conf=conf_np, th=self.conf_th)

                graph_dir = f"{self.output_dir}/graph"
                self.graph.visualize_edges(t1+1, save_path=graph_dir, selected_nodes=i-3)

    
    def run(self, tstamp, last_frame=False):        
        # do initialization
        if not self.keyframes.is_initialized and self.keyframes.counter.value - 1 == self.warmup:
            t1 = self.keyframes.counter.value - 1
            self.track(0, t1, init=True)
            self.keyframes.is_initialized = True
            self.t1 = t1      

            if self.verbose:
                sub_num = 0
                pts = (self.keyframes.submap_ds[:sub_num + 1].flatten(0, 1))[:t1-1].cpu().numpy()
                confs = (self.keyframes.conf_ds[:sub_num + 1].flatten(0, 1))[:t1-1].cpu().numpy()
                images = self.keyframes.image[:t1-1].permute(0, 2, 3, 1).cpu().numpy() / 255.0
                images = images[:, ::self.downsample_ratio, ::self.downsample_ratio]
                viz_pcd(pts, images, os.path.join(self.output_dir, "pcd"), self.keyframes.tstamp[t1-1], conf=confs, th=self.conf_th)

            return False, range(0, t1), 0
            
        elif self.keyframes.is_initialized and self.t1 < self.keyframes.counter.value - 5:
            t0 = self.t1 - 1
            t1 = self.keyframes.counter.value - 1
            self.track(t0, t1)
            self.t1 = t1

            if self.verbose:
                sub_num = t0 // 5
                pts = (self.keyframes.submap_ds[:sub_num + 1, :-1].flatten(0, 1))[:t1-1].cpu().numpy()
                confs = (self.keyframes.conf_ds[:sub_num + 1, :-1].flatten(0, 1))[:t1-1].cpu().numpy()
                images = self.keyframes.image[:t1-1].permute(0, 2, 3, 1).cpu().numpy() / 255.0
                images = images[:, ::self.downsample_ratio, ::self.downsample_ratio]
                viz_pcd(pts, images, os.path.join(self.output_dir, "pcd"), self.keyframes.tstamp[t1-1], conf=confs, th=self.conf_th)

            if t1 > 10:
                return True, range(t0, t1), t0 // 5
            else:
                return False, range(t0, t1), t0 // 5
            
        elif last_frame:
            t0 = self.t1 - 1
            t1 = self.keyframes.counter.value - 1
            self.track(t0, t1)
            self.t1 = t1

            return False, range(t0, t1), t0 // 5
        else:
            return False, None, None

    
    def test(self, tstamp, last_frame=False):        
        # do initialization
        if not self.keyframes.is_initialized and self.keyframes.counter.value - 1 == self.warmup:
            t1 = self.keyframes.counter.value - 1
            for i in range(0, t1):
                depth = self.keyframes.depth[i].to(self.device)
                T_c2w = pose_vec_to_matrix(self.keyframes.pose[i].unsqueeze(0).to(self.device))
                intrinsic = self.keyframes.intrinsic[0].to(self.device)
                fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
                pointmap = depth_to_pointmap(depth[None], T_c2w, fx, fy, cx, cy)[0]
                self.keyframes.submap_ds[0, i] = pointmap[::self.downsample_ratio, ::self.downsample_ratio]
                self.keyframes.conf_ds[0, i] = torch.where(self.keyframes.depth[i] > 1.0, 1.0, 0.0)[::self.downsample_ratio, ::self.downsample_ratio]
                self.keyframes.pose[i] = perturb_pose(self.keyframes.pose[i], trans_sigma=0.05, rot_sigma=0.01)

            self.keyframes.is_initialized = True
            self.t1 = t1      

            if self.verbose:
                sub_num = 0
                pts = (self.keyframes.submap_ds[:sub_num + 1].flatten(0, 1))[:t1-1].cpu().numpy()
                confs = (self.keyframes.conf_ds[:sub_num + 1].flatten(0, 1))[:t1-1].cpu().numpy()
                images = self.keyframes.image[:t1-1].permute(0, 2, 3, 1).cpu().numpy() / 255.0
                images = images[:, ::self.downsample_ratio, ::self.downsample_ratio]
                viz_pcd(pts, images, os.path.join(self.output_dir, "pcd"), self.keyframes.tstamp[t1-1], conf=confs, th=self.conf_th)

            return False, range(0, t1), 0
            
        elif self.keyframes.is_initialized and self.t1 < self.keyframes.counter.value - 5:
            t0 = self.t1 - 1
            t1 = self.keyframes.counter.value - 1
            for i in range(t0, t1):
                depth = self.keyframes.depth[i].to(self.device)
                T_c2w = pose_vec_to_matrix(self.keyframes.pose[i].unsqueeze(0).to(self.device))
                intrinsic = self.keyframes.intrinsic[0].to(self.device)
                fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
                pointmap = depth_to_pointmap(depth[None], T_c2w, fx, fy, cx, cy)[0]
                self.keyframes.submap_ds[t0//5, i-t0] = pointmap[::self.downsample_ratio, ::self.downsample_ratio]
                self.keyframes.conf_ds[t0//5, i-t0] += 1
                self.keyframes.pose[i] = perturb_pose(self.keyframes.pose[i], trans_sigma=0.05, rot_sigma=0.01)

            self.t1 = t1

            if self.verbose:
                sub_num = t0 // 5
                pts = (self.keyframes.submap_ds[:sub_num + 1, :-1].flatten(0, 1))[:t1-1].cpu().numpy()
                confs = (self.keyframes.conf_ds[:sub_num + 1, :-1].flatten(0, 1))[:t1-1].cpu().numpy()
                images = self.keyframes.image[:t1-1].permute(0, 2, 3, 1).cpu().numpy() / 255.0
                images = images[:, ::self.downsample_ratio, ::self.downsample_ratio]
                viz_pcd(pts, images, os.path.join(self.output_dir, "pcd"), self.keyframes.tstamp[t1-1], conf=confs, th=self.conf_th)

            if t1 > 10:
                return True, range(t0, t1), t0 // 5
            else:
                return False, range(t0, t1), t0 // 5
            
        elif last_frame:
            t0 = self.t1 - 1
            t1 = self.keyframes.counter.value - 1
            for i in range(t0, t1):
                depth = self.keyframes.depth[i].to(self.device)
                T_c2w = pose_vec_to_matrix(self.keyframes.pose[i].unsqueeze(0).to(self.device))
                intrinsic = self.keyframes.intrinsic[0].to(self.device)
                fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
                pointmap = depth_to_pointmap(depth[None], T_c2w, fx, fy, cx, cy)[0]
                self.keyframes.submap_ds[t0//5, i-t0] = pointmap[::self.downsample_ratio, ::self.downsample_ratio]
                self.keyframes.conf_ds[t0//5, i-t0] += 1
                self.keyframes.pose[i] = perturb_pose(self.keyframes.pose[i], trans_sigma=0.05, rot_sigma=0.01)
                
            self.t1 = t1

            return False, range(t0, t1), t0 // 5
        else:
            return False, None, None
        
import torch.nn.functional as F
def perturb_pose(pose, trans_sigma=0.001, rot_sigma=0.001):
    trans = pose[:3]
    quat = pose[3:]

    trans_noise = torch.randn_like(trans) * trans_sigma
    trans_perturbed = trans + trans_noise

    quat_noise = torch.randn_like(quat) * rot_sigma
    quat_perturbed = quat + quat_noise
    quat_perturbed = F.normalize(quat_perturbed, dim=-1)

    pose_perturbed = torch.cat([trans_perturbed, quat_perturbed], dim=-1)
    return pose_perturbed
    # return pose