import random
import time
import os
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from munch import munchify
from lietorch import SE3, SO3
import torchvision
import torch.nn.functional as F
import open3d as o3d
from gaussian.renderer import render
from gaussian.utils.loss_utils import l1_loss, ssim
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.graphics_utils import getProjectionMatrix2
from gaussian.utils.slam_utils import update_pose, get_pose, depth_to_normal, apply_depth_colormap, project2world
from gaussian.utils.camera_utils import Camera
from gaussian.utils.eval_utils import eval_rendering, eval_rendering_kf
from gaussian.gui import gui_utils#, slam_gui
from scipy.spatial.transform import Rotation
from util.utils import Log, pose_vec_to_matrix, sobel_edges, gaussian_blur

class GSBackEnd(mp.Process):
    def __init__(self, slam, config, save_dir, use_gui=False):
        super().__init__()
        self.config = config
        
        self.iteration_count = 0
        self.viewpoints = {}
        self.current_window = []
        self.initialized = False
        self.save_dir = save_dir
        self.use_gui = use_gui

        self.opt_params = munchify(config["opt_params"])
        self.lambda_depth = self.config['Training']['lambda_depth']
        self.lambda_tv = self.config['Training']['lambda_tv']
        self.lambda_iso = self.config['Training']['lambda_iso']
        self.lambda_normal = self.config['Training']['lambda_normal']

        self.downsample_ratio = slam.downsample_ratio
        self.output_dir = slam.output_dir
        self.verbose = slam.verbose

        self.gaussians = GaussianModel(sh_degree=0, config=self.config)
        self.gaussians.init_lr(1.0)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        self.cameras_extent = 6.0
        self.set_hyperparams()

        if self.use_gui:
            self.q_main2vis = mp.Queue()
            self.q_vis2main = mp.Queue()
            self.params_gui = gui_utils.ParamsGUI(
                background=self.background,
                gaussians=self.gaussians,
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(3)

    def set_hyperparams(self):
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]

        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = self.cameras_extent * self.config["Training"]["gaussian_extent"]
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]

        self.mapping_update_every = self.config["Mapping"]["gaussian_update_every"]
        self.mapping_update_offset = self.config["Mapping"]["gaussian_update_offset"]

    def reset(self):
        self.iteration_count = 0
        self.current_window = []
        self.initialized = False
        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

    def add_new_view(self, new_img, new_pose, new_depth, new_pointmap, new_conf, new_tstamp, kf_sub_idx, iterations=100):     
        H, W = new_img.shape[-2:]   

        imgs =new_img.cuda() / 255.0
        
        pointmaps = new_pointmap
        depths = new_depth
        c2w = pose_vec_to_matrix(new_pose).cuda()
        w2c = torch.inverse(c2w)

        imgs_ds = imgs[..., ::self.downsample_ratio, ::self.downsample_ratio]
        pointmaps = F.interpolate(pointmaps.permute(0, 3, 1, 2), size=(H//self.downsample_ratio, W//self.downsample_ratio), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).numpy()
        depths = F.interpolate(depths[None], size=(H, W), mode='bilinear', align_corners=False)[0]
        _, h1, w1, _ = pointmaps.shape

        tstamp = new_tstamp
        idx = len(self.viewpoints)
        viewpoint = Camera.init_from_tracking(imgs[0], depths[0], None, w2c[0], idx, self.projection_matrix, self.K, tstamp, None)
        self.viewpoints[idx] = viewpoint

        current_idx = [idx]
        
        pointmap, conf = self.pose_refine(current_idx, self.fx, self.fy, self.cx, self.cy, iters=50)
        pointmap = pointmap.detach().cpu().numpy()
        conf = conf.detach().cpu().numpy()
        rgb = imgs_ds[0][None]
    
        self.gaussians.extend_from_pcd_seq(rgb=rgb, 
                                        pointmap=pointmap,
                                        normal=None, 
                                        conf=conf, 
                                        submap_idx=kf_sub_idx, init=False)                    
        
        self.optimization(20, current_window=current_idx, optimize_pose=False)
        

    def pose_estimator(self, pose, gt_img, tstamp, gt_depth=None, confs=None):  
        c2w = pose_vec_to_matrix(pose[None]).squeeze(0).cuda()
        w2c = torch.inverse(c2w)
        viewpoint = Camera.init_from_tracking(gt_img/255.0, gt_depth, None, w2c, 0, self.projection_matrix, self.K, tstamp, confs)

        opt_params = []
        opt_params.append({
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["opt_params"]["pose_lr"],
                "name": "rot_{}".format(viewpoint.uid)})
        opt_params.append({
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["opt_params"]["pose_lr"],
                "name": "trans_{}".format(viewpoint.uid)})
            
        pose_optimizers = torch.optim.Adam(opt_params)

        self.gaussians.optimizer.zero_grad(set_to_none=True)
        pose_optimizers.zero_grad(set_to_none=True)

        for iteration in range(100):
            # gaussian alignment: fix the gaussians and render for new viewpoint, mask out area without gaussians
            current_w2c = get_pose(viewpoint)

            render_pkg = render(viewpoint, self.gaussians, self.background)

            image, depth = (render_pkg["render"], render_pkg["depth"])

            gt_image = viewpoint.original_image.cuda()
            rgb_loss = 0.8 * torch.abs((gt_image - image)).mean() + 0.2 * (1.0 - ssim(image, gt_image))
            loss = rgb_loss
            
            if gt_depth is not None:
                gt_depth = viewpoint.depth.to(dtype=torch.float32, device=image.device)[None]
                depth_pixel_mask = torch.logical_and(gt_depth > 0.001, depth > 0.001).view(*depth.shape)
                depth_loss = torch.abs(1./depth[depth_pixel_mask] - 1./gt_depth[depth_pixel_mask]).mean()
                loss += depth_loss

            loss.backward()
            pose_optimizers.step()
            pose_optimizers.zero_grad(set_to_none=True)
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():                    
                update_pose(viewpoint)

        with torch.no_grad(): 
            current_w2c = get_pose(viewpoint)
            current_c2w = torch.inverse(current_w2c).cpu()
            
            quat = torch.from_numpy(Rotation.from_matrix(current_c2w[:3, :3]).as_quat())
            trans = current_c2w[:3, 3]
            updated_pose = torch.cat([trans, quat], dim=-1)

        return updated_pose
   
    def prefilter(self, fliter_window):
        '''filer out where gaussians need to be added'''
        with torch.no_grad(): 
            viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in fliter_window]
            i= 0
            alpha = []
            for viewpoint in viewpoint_stack:
                render_pkg = render(viewpoint, self.gaussians, self.background)   
                alpha_mask = render_pkg["mask"]

                alpha.append(alpha_mask)
                
                if self.verbose:
                    kf_idx = fliter_window[i]  
                    os.makedirs(f"{self.output_dir}/prefilter", exist_ok = True)
                    self.viz(viewpoint, f"{self.output_dir}/prefilter", kf_idx)
                    i += 1
            
            alpha = torch.cat(alpha)
            alpha_binary = (alpha <= 0.9).float()

        return alpha_binary
    
    def pose_refine(self, BA_window, fx, fy, cx, cy, iters=50, lr_ratio=10, return_args=True, alpha_th=0.9):  
        '''
            optimize pose jointly with gaussian model and reprojection
        '''      
        viewpoints = [self.viewpoints[kf_idx] for kf_idx in BA_window]
        B = len(viewpoints)

        opt_params = []
        for idx, view in enumerate(viewpoints):
            opt_params.append({
                    "params": [view.cam_rot_delta],
                    "lr": self.config["opt_params"]["pose_lr"]*2,
                    "name": "rot_{}".format(view.uid)})
            opt_params.append({
                    "params": [view.cam_trans_delta],
                    "lr": self.config["opt_params"]["pose_lr"]*10,
                    "name": "trans_{}".format(view.uid)})
                
        pose_optimizers = torch.optim.Adam(opt_params)

        self.gaussians.optimizer.zero_grad(set_to_none=True)
        pose_optimizers.zero_grad(set_to_none=True)

        # pbar = tqdm(range(iters), desc="pose refinement")
        for iteration in range(iters):
            w2c_all = []
            gt_image_all = []
            alpha_all = []
            
            rgb_loss_all = 0
            depth_loss_all = 0
            for viewpoint in viewpoints:
                # gaussian alignment: fix the gaussians and render for new viewpoint, mask out area without gaussians
                current_w2c = get_pose(viewpoint)

                render_pkg = render(viewpoint, self.gaussians, self.background)

                image, depth, alpha = (render_pkg["render"], render_pkg["depth"], render_pkg["mask"])

                gt_image = viewpoint.original_image.cuda()
                gt_depth = viewpoint.depth.to(dtype=torch.float32, device=image.device)[None]

                alpha_mask = (alpha > alpha_th).detach()
                existing_ratio = alpha_mask.sum() / (alpha_mask.shape[1] * alpha_mask.shape[2])
                if existing_ratio > 0.1:
                    depth_pixel_mask = torch.logical_and(gt_depth > 0.001, depth > 0.001).view(*depth.shape)
                    depth_pixel_mask = torch.logical_and(depth_pixel_mask, alpha_mask)

                    rgb_loss = torch.abs((gt_image - image)[:, alpha_mask[0]]).mean()
                    # img_smooth = gaussian_blur(image)
                    # gt_smooth = gaussian_blur(gt_image)
                    # edge_loss = torch.abs((sobel_edges(img_smooth) - sobel_edges(gt_smooth))).mean()

                    depth_loss = torch.abs(1./depth[depth_pixel_mask] - 1./gt_depth[depth_pixel_mask]).mean()
                    # diff = torch.log(depth[depth_pixel_mask]) - torch.log(gt_depth[depth_pixel_mask])
                    # depth_loss = (diff**2).mean() - diff.mean()**2
                    
                    rgb_loss_all += rgb_loss #0.8 * rgb_loss + 0.2 * edge_loss
                    depth_loss_all +=  depth_loss

                w2c_all.append(current_w2c)
                gt_image_all.append(gt_image.permute(1, 2, 0))
                alpha_all.append(alpha_mask)
            
            alpha_all = torch.cat(alpha_all)
            alpha_binary = (alpha_all <= alpha_th).float()

            loss = (rgb_loss_all + depth_loss_all) / B

            loss.backward()
            
            # pbar.set_postfix(loss=f"{loss.item():.4f}")
            # pbar.update()

            with torch.no_grad():                    
                pose_optimizers.step()
                pose_optimizers.zero_grad(set_to_none=True)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                
                for viewpoint in viewpoints:
                    update_pose(viewpoint)

        w2c_all = []
        gt_depth_all = []
        if return_args:
            with torch.no_grad(): 
                for i, viewpoint in enumerate(viewpoints):
                    current_w2c = get_pose(viewpoint)
                    gt_depth = viewpoint.depth.to(dtype=torch.float32, device=image.device)[None]

                    render_pkg = render(viewpoint, self.gaussians, self.background)
                    image, depth, alpha = (render_pkg["render"], render_pkg["depth"], render_pkg["mask"])

                    alpha_mask = (alpha > alpha_th)
                    depth_pixel_mask = torch.logical_and(gt_depth > 0.001, depth > 0.001).view(*depth.shape)
                    depth_pixel_mask = torch.logical_and(depth_pixel_mask, alpha_mask)
                    depth_valid = depth[depth_pixel_mask]
                    gt_valid = gt_depth[depth_pixel_mask]

                    if depth_valid.numel() > 10:
                        log_scale = (torch.log(depth_valid) - torch.log(gt_valid)).mean()
                        scale = torch.exp(log_scale)
                        scale = torch.clamp(scale, 0.95, 1.05)
                        gt_depth = scale * gt_depth

                    w2c_all.append(current_w2c)
                    gt_depth_all.append(gt_depth)

                    if self.verbose:
                        kf_idx = BA_window[i]  
                        os.makedirs(f"{self.output_dir}/pre_BA", exist_ok = True)
                        self.viz(viewpoint, f"{self.output_dir}/pre_BA", kf_idx)

                T_w2c = torch.stack(w2c_all, dim=0)
                gt_depths = torch.cat(gt_depth_all, dim=0)

                # return pointmaps corresponding to updated pose
                pointmaps = project2world(torch.inverse(T_w2c), gt_depths, fx, fy, cx, cy)

            return pointmaps[:, ::self.downsample_ratio, ::self.downsample_ratio], alpha_binary[:, ::self.downsample_ratio, ::self.downsample_ratio]


    def pre_optimization(self, BA_window, iters=20, alpha_th=0.9, densify=True):  
        viewpoints = [self.viewpoints[kf_idx] for kf_idx in BA_window]
        B = len(viewpoints)

        opt_params = []
        view = viewpoints[-1]
        opt_params.append({
                "params": [view.cam_rot_delta],
                "lr": self.config["opt_params"]["pose_lr"]*2,
                "name": "rot_{}".format(view.uid)})
        opt_params.append({
                "params": [view.cam_trans_delta],
                "lr": self.config["opt_params"]["pose_lr"]*10,
                "name": "trans_{}".format(view.uid)})
        
        pose_optimizers = torch.optim.Adam(opt_params)

        self.gaussians.optimizer.zero_grad(set_to_none=True)
        pose_optimizers.zero_grad(set_to_none=True)

        pbar = tqdm(range(iters), desc="Pre Optimization")
        for iteration in range(iters):            
            rgb_loss_all = 0
            depth_loss_all = 0
            normal_loss_all = 0
            isotropic_loss_all = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            for idx, viewpoint in enumerate(viewpoints):
                render_pkg = render(viewpoint, self.gaussians, self.background)

                alpha = render_pkg["mask"]
                image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"])

                gt_image = viewpoint.original_image.cuda()
                gt_depth = viewpoint.depth.to(dtype=torch.float32, device=image.device)[None]

                alpha_mask = (alpha > alpha_th).detach()
                rgb_loss = torch.abs((gt_image - image) * alpha_mask).mean()

                depth_pixel_mask = torch.logical_and(gt_depth > 0.001, depth > 0.001).view(*depth.shape).detach()
                depth_pixel_mask = torch.logical_and(depth_pixel_mask, alpha_mask)
                depth_loss = torch.abs((1./depth[depth_pixel_mask] - 1./gt_depth[depth_pixel_mask])).mean()
                # diff = torch.log(depth[depth_pixel_mask]) - torch.log(gt_depth[depth_pixel_mask])
                # depth_loss = (diff**2).mean() - diff.mean()**2

                depth_normal, _ = depth_to_normal(viewpoint, depth, world_frame=False)
                depth_normal = depth_normal.permute(2, 0, 1)
                nan_mask = torch.isnan(depth_normal)
                depth_normal = depth_normal * alpha_mask * (~nan_mask)

                gt_depth_normal, _ = depth_to_normal(viewpoint, gt_depth.detach(), world_frame=False)
                gt_depth_normal = gt_depth_normal.permute(2, 0, 1)
                gt_nan_mask = torch.isnan(gt_depth_normal)
                gt_depth_normal = gt_depth_normal * alpha_mask * (~gt_nan_mask).detach()
                normal_loss = (1 - (depth_normal * gt_depth_normal).sum(dim=0)).mean()
                
                scaling = self.gaussians.get_scaling[visibility_filter]
                isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean()

                rgb_loss_all += rgb_loss
                depth_loss_all +=  depth_loss
                normal_loss_all += normal_loss
                isotropic_loss_all += isotropic_loss
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            loss = (rgb_loss_all + 
                    self.lambda_depth * depth_loss_all + 
                    self.lambda_normal * normal_loss_all +
                    self.lambda_iso * isotropic_loss_all) / B

            loss.backward()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update()

            with torch.no_grad():
                if densify:
                    for idx in range(len(viewspace_point_tensor_acm)):
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                            self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                            radii_acm[idx][visibility_filter_acm[idx]],
                        )
                        self.gaussians.add_densification_stats(
                            viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                        )

                    if iteration == iters // 2:
                        self.gaussians.densify_and_prune(
                            self.opt_params.densify_grad_threshold,
                            self.gaussian_th,
                            self.gaussian_extent,
                            self.size_threshold
                        )
             
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                pose_optimizers.step()
                pose_optimizers.zero_grad(set_to_none=True)
                
                update_pose(viewpoints[-1])

        with torch.no_grad(): 
            os.makedirs(f"{self.output_dir}/pre_optimization", exist_ok = True)
            self.viz(viewpoints[-1], f"{self.output_dir}/pre_optimization", BA_window[-1])
            

    def optimization(self, iters, optimize_pose=True, current_window=None):
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        B = len(viewpoint_stack)

        if optimize_pose:
            opt_params = []
            for idx, view in enumerate(viewpoint_stack):
                opt_params.append({
                        "params": [view.cam_rot_delta],
                        "lr": self.config["opt_params"]["pose_lr"]*2,
                        "name": "rot_{}".format(view.uid)})
                opt_params.append({
                        "params": [view.cam_trans_delta],
                        "lr": self.config["opt_params"]["pose_lr"]*10,
                        "name": "trans_{}".format(view.uid)})
                    
            pose_optimizers = torch.optim.Adam(opt_params)

            pose_optimizers.zero_grad(set_to_none=True)
        self.gaussians.optimizer.zero_grad(set_to_none=True)        

        current_window_set = set(current_window)
        random_viewpoint_stack = []
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx not in current_window_set:
                random_viewpoint_stack.append(viewpoint)

        pbar = tqdm(range(iters), desc="Gaussian Mapping")
        for iteration in range(iters):
            self.iteration_count += 1

            rgb_loss_all = 0
            depth_loss_all = 0
            normal_loss_all = 0
            isotropic_loss_all = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            gt_depth_all = []
            w2c_all = []
            gt_image_all = []

            viewpoints = viewpoint_stack + [random_viewpoint_stack[idx] for idx in torch.randperm(len(random_viewpoint_stack))[:2]]
            N = len(viewpoints)
            for viewpoint in viewpoints:
                rel_w2c = get_pose(viewpoint)
                    
                render_pkg = render(viewpoint, self.gaussians, self.background)

                image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"])

                gt_image = viewpoint.original_image.cuda()
                gt_depth = viewpoint.depth.to(dtype=torch.float32, device=image.device)[None]

                rgb_loss = 0.8 * torch.abs((gt_image - image)).mean() + 0.2 * (1.0 - ssim(image, gt_image))

                depth_pixel_mask = torch.logical_and(gt_depth > 0.001, depth > 0.001).view(*depth.shape).detach()
                depth_loss = torch.abs(1./depth[depth_pixel_mask] - 1./gt_depth[depth_pixel_mask]).mean()
                # diff = torch.log(depth[depth_pixel_mask]) - torch.log(gt_depth[depth_pixel_mask])
                # depth_loss = (diff**2).mean() - diff.mean()**2

                rendered_normal = render_pkg["normal"]
                depth_normal, _ = depth_to_normal(viewpoint, depth, world_frame=False)
                depth_normal = depth_normal.permute(2, 0, 1)
                # normal_loss = (1 - (rendered_normal * depth_normal).sum(dim=0)).mean()

                gt_depth_normal, _ = depth_to_normal(viewpoint, gt_depth.detach(), world_frame=False)
                gt_depth_normal = gt_depth_normal.permute(2, 0, 1)
                normal_loss = (1 - (depth_normal * gt_depth_normal).sum(dim=0)).mean()
                
                scaling = self.gaussians.get_scaling[visibility_filter]
                isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean()

                rgb_loss_all += rgb_loss
                depth_loss_all +=  depth_loss
                normal_loss_all += normal_loss
                isotropic_loss_all += isotropic_loss
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                gt_depth_all.append(gt_depth)
                w2c_all.append(rel_w2c)
                gt_image_all.append(gt_image.permute(1, 2, 0))

            loss = (rgb_loss_all + 
                    self.lambda_depth * depth_loss_all + 
                    self.lambda_normal * normal_loss_all +
                    self.lambda_iso * isotropic_loss_all) / N

            loss.backward()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update()

            self.gaussians.optimizer.step()
            
            if optimize_pose:
                pose_optimizers.step()
                with torch.no_grad():
                    for viewpoint in viewpoint_stack:
                        update_pose(viewpoint)   
                pose_optimizers.zero_grad(set_to_none=True)
            self.gaussians.optimizer.zero_grad(set_to_none=True)

        if self.verbose:
            with torch.no_grad():
                for i, viewpoint in enumerate(viewpoint_stack):
                    kf_idx = current_window[i]  
                    os.makedirs(f"{self.output_dir}/rendering", exist_ok = True)
                    self.viz(viewpoint, f"{self.output_dir}/rendering", kf_idx)


    def viz(self, viewpoint, dir, idx):
        render_pkg = render(viewpoint, self.gaussians, self.background)
            
        image, depth = (render_pkg["render"], render_pkg["depth"])  
        
        mask = (render_pkg["mask"] > 0.9).float()
        mask_map = apply_depth_colormap(mask.permute(1, 2, 0), None, near_plane=None, far_plane=None)
        mask_map = mask_map.permute(2, 0, 1)

        normal, _ = depth_to_normal(viewpoint, depth, world_frame=False)
        normal = (normal + 1.) / 2.
        normal = normal.permute(2, 0, 1)

        render_normal = render_pkg["normal"]
        render_normal = (render_normal + 1.) / 2.
        
        depth_map = apply_depth_colormap(depth.permute(1, 2, 0), None, near_plane=None, far_plane=None)
        depth_map = depth_map.permute(2, 0, 1)

        gt_image = viewpoint.original_image.cuda()
        image_ = (image.permute(1, 2, 0) @ viewpoint.exposure_a + viewpoint.exposure_b).permute(2, 0, 1)

        gt_depth = viewpoint.depth.to(dtype=torch.float32, device=image.device)[..., None]

        gt_depth_map = apply_depth_colormap(gt_depth, None, near_plane=None, far_plane=None)
        gt_depth_map = gt_depth_map.permute(2, 0, 1)

        depth_diff = torch.abs(gt_depth - depth.permute(1, 2, 0))
        depth_diff = apply_depth_colormap(depth_diff, None, near_plane=None, far_plane=None)
        depth_diff = depth_diff.permute(2, 0, 1)

        row0 = torch.cat([gt_image, image, torch.abs(gt_image - image_)*10], dim=2)
        row1 = torch.cat([gt_depth_map, depth_map, depth_diff], dim=2)
        
        image_to_show = torch.cat([row0, row1], dim=1)
        
        row2 = torch.cat([render_normal, normal, mask_map], dim=2)
        image_to_show = torch.cat([row0, row1, row2], dim=1)
            
        image_to_show = torch.clamp(image_to_show, 0, 1)
        tstamp = viewpoint.tstamp
        torchvision.utils.save_image(image_to_show, f"{dir}/{idx}_tstamp_{tstamp}.jpg")

    def pcd_viz(self, idx):
        pts = self.gaussians.get_xyz.detach().cpu().numpy()
        colors = (self.gaussians._features_dc.detach().cpu().numpy()) * 0.28209479177387814 + 0.5

        os.makedirs(f'{self.output_dir}', exist_ok=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
        o3d.io.write_point_cloud(f'{self.output_dir}/gaussians_viz_{idx}.ply', pcd)

    def data_update(self, h, w, current_window):
        '''update pointmaps, depths, normals and poses for keyframes'''
        with torch.no_grad():
            viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
            poses = []
            depths = []
            # normals = []
            pointmaps = []
            for view in viewpoint_stack:       
                gt_depth = view.depth[None].to(view.device)             
                render_pkg = render(view, self.gaussians, self.background)
                depth = render_pkg["depth"]
                alpha = render_pkg["mask"]

                valid_mask = (depth > 0.001) & (gt_depth > 0.001) & (alpha > 0.9)
                depth_valid = depth[valid_mask]
                gt_valid = gt_depth[valid_mask]
                log_scale = (torch.log(depth_valid) - torch.log(gt_valid)).mean()
                scale = torch.exp(log_scale)
                scale = torch.clamp(scale, 0.95, 1.05)
                gt_depth = scale * gt_depth
                view.depth = gt_depth.squeeze(0).cpu()

                _, pointmap = depth_to_normal(view, gt_depth, world_frame=True)
                depth = gt_depth

                T_w2c = np.eye(4)
                T_w2c[0:3, 0:3] = view.R.cpu().numpy()
                T_w2c[0:3, 3] = view.T.cpu().numpy()
                T_c2w = torch.inverse(torch.from_numpy(T_w2c))
                r = Rotation.from_matrix(T_c2w[:3, :3])
                quat = torch.from_numpy(r.as_quat())
                trans = T_c2w[:3, 3]

                depths.append(depth[0])
                # normals.append(normal)
                pointmaps.append(pointmap)
                poses.append(torch.cat([trans, quat], dim=-1))

            pointmaps = torch.stack(pointmaps)
            pointmaps = F.interpolate(pointmaps.permute(0, 3, 1, 2), size=(h * self.downsample_ratio, w * self.downsample_ratio), mode='bilinear', align_corners=False)
            pointmaps = pointmaps.permute(0, 2, 3, 1)
            depths = torch.stack(depths)
            depths = F.interpolate(depths[None], size=(h * self.downsample_ratio, w * self.downsample_ratio), mode='bilinear', align_corners=False)[0]

            updated_packet ={
                'pointmaps':  pointmaps,
                'depths':   depths,
                'poses':  torch.stack(poses)}
            
            return updated_packet, current_window
            
    def gaussain_update(self, packet):
        with torch.no_grad():
            camera_ids = packet['camera_idx']
            c2ws = pose_vec_to_matrix(packet["camera_pose"]).cuda()
            w2cs = torch.inverse(c2ws)
            update_idx = []
            for i, camera_idx in enumerate(camera_ids):
                if camera_idx in self.viewpoints.keys():
                    update_idx.append(camera_idx)
                    w2c = w2cs[i]
                    self.viewpoints[camera_idx].R = w2c[:3, :3]
                    self.viewpoints[camera_idx].T = w2c[:3, 3]
                    self.viewpoints[camera_idx].R_gt = w2c[:3, :3]
                    self.viewpoints[camera_idx].T_gt = w2c[:3, 3]

            submap_idx = torch.from_numpy(np.array(packet['submap_idx']))
            gaussian_indices = (self.gaussians.unique_kfIDs.unsqueeze(1) == submap_idx.unsqueeze(0)).nonzero()[:, 0]

            indices = (submap_idx.unsqueeze(1) == self.gaussians.unique_kfIDs.unsqueeze(0)).nonzero()[:, 0]
            
            poses = packet['pose_updates'].cuda()[indices]
            se3 = SE3(poses) 
            T = se3.matrix()
            align_R = T[:, :3, :3]
            align_t = T[:, :3, 3].unsqueeze(1)
                
            xyz = self.gaussians.get_xyz[gaussian_indices]
            new_xyz = (torch.matmul(xyz.unsqueeze(1), align_R.transpose(1, 2)) + align_t).squeeze(1)

            rot = SO3(self.gaussians.get_rotation[gaussian_indices])
            new_rotation = (SO3(poses[:,3:]) * rot).data

            new_features_dc = self.gaussians._features_dc[gaussian_indices]
            new_features_rest = self.gaussians._features_rest[gaussian_indices]
            new_opacity = self.gaussians._opacity[gaussian_indices]
            new_scaling = self.gaussians._scaling[gaussian_indices]
            new_kfIDs = self.gaussians.unique_kfIDs[gaussian_indices]
            new_n_obs = self.gaussians.n_obs[gaussian_indices]

            prune_mask = torch.zeros(self.gaussians.get_xyz.shape[0], dtype=bool).cuda()
            prune_mask[gaussian_indices] = True
            self.gaussians.prune_points(prune_mask)

            d = {
                "xyz": new_xyz,
                "f_dc": new_features_dc,
                "f_rest": new_features_rest,
                "opacity": new_opacity,
                "scaling": new_scaling,
                "rotation": new_rotation,
            }

            optimizable_tensors = self.gaussians.cat_tensors_to_optimizer(d)
            self.gaussians._xyz = optimizable_tensors["xyz"]
            self.gaussians._features_dc = optimizable_tensors["f_dc"]
            self.gaussians._features_rest = optimizable_tensors["f_rest"]
            self.gaussians._opacity = optimizable_tensors["opacity"]
            self.gaussians._scaling = optimizable_tensors["scaling"]
            self.gaussians._rotation = optimizable_tensors["rotation"]

            self.gaussians.xyz_gradient_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
            self.gaussians.xyz_gradient_accum_abs = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
            self.gaussians.xyz_gradient_accum_abs_max = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
            self.gaussians.denom = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
            self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")
            self.gaussians.unique_kfIDs = torch.cat((self.gaussians.unique_kfIDs, new_kfIDs)).int()
            self.gaussians.n_obs = torch.cat((self.gaussians.n_obs, new_n_obs)).int()
  
        for idx in update_idx:
            self.pose_refine([idx], self.fx, self.fy, self.cx, self.cy, iters=50, return_args=False, alpha_th=0.5)

        return self.data_update(self.h, self.w, update_idx)

    def run(self, packet, iterations=100):     
        H, W = packet["images"].shape[-2:]   
        if not hasattr(self, "projection_matrix"):
            self.K = K = list(packet["intrinsics"]) + [W, H]
            self.fx=K[0]
            self.fy=K[1]
            self.cx=K[2]
            self.cy=K[3]
            self.projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=K[0], fy=K[1], cx=K[2], cy=K[3], W=W, H=H).transpose(0, 1).cuda()

        viz_idx = packet['viz_idx']
        submap_idx = packet['submap_idx']
        imgs = packet['images'].cuda() / 255.0
        
        pointmaps = packet['pointmaps']
        depths = packet["depths"]
        confs = packet['confs'].numpy()
        c2w = pose_vec_to_matrix(packet["poses"]).cuda()
        w2c = torch.inverse(c2w)

        _, self.h, self.w, _ = pointmaps.shape
        imgs_ds = imgs[..., ::self.downsample_ratio, ::self.downsample_ratio]
        pointmaps = F.interpolate(pointmaps.permute(0, 3, 1, 2), size=(H//self.downsample_ratio, W//self.downsample_ratio), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).numpy()
        depths = F.interpolate(depths[None], size=(H, W), mode='bilinear', align_corners=False)[0]
        _, h1, w1, _ = pointmaps.shape

        for i, idx in enumerate(viz_idx):
            if idx not in self.viewpoints.keys():
                tstamp = packet['tstamp'][i].item()
                viewpoint = Camera.init_from_tracking(imgs[i], depths[i], None, w2c[i], idx, self.projection_matrix, self.K, tstamp, None)
                self.viewpoints[idx] = viewpoint
        
                # add new gaussian per frame, mapping and joint optimization gaussians and poses
                if not self.initialized:
                    self.reset()
                    rgb = imgs_ds[0][None]
                    pointmap = pointmaps[0][None]
                    conf = confs[0][None]
                    self.gaussians.extend_from_pcd_seq(rgb=rgb, 
                                                    pointmap=pointmap, 
                                                    normal=None, 
                                                    conf=None, 
                                                    submap_idx=0, init=True)
                    self.current_window = list([0])
                    self.optimization(100, current_window=self.current_window, optimize_pose=False)
                    self.initialized = True
                else:  # add new kf into gaussian model 
                    current_idx = [idx]
                    
                    if len(self.current_window) < self.window_size:
                        self.current_window = self.current_window + list(current_idx)
                    else:
                        self.current_window = self.current_window[1:] + list(current_idx)
                    self.pre_optimization(self.current_window[-2:], iters=100, densify=False)
                    
                    pointmap, conf = self.pose_refine(current_idx, self.fx, self.fy, self.cx, self.cy, iters=1)
                    # pointmap = F.interpolate(pointmap.permute(0, 3, 1, 2), size=(h1, w1), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    pointmap = pointmap.detach().cpu().numpy()
                    # conf = F.interpolate(conf[None], size=(h1, w1), mode='bilinear', align_corners=False)[0]
                    conf = conf.detach().cpu().numpy()
                    rgb = imgs_ds[i][None]
                
                    self.gaussians.extend_from_pcd_seq(rgb=rgb, 
                                                    pointmap=pointmap,
                                                    normal=None, 
                                                    conf=conf, 
                                                    submap_idx=submap_idx, init=False)                    
                    
                    self.optimization(20, current_window=self.current_window)
        
        for idx in self.current_window:
            self.pose_refine([idx], self.fx, self.fy, self.cx, self.cy, iters=10, return_args=False, alpha_th=0.5)
        
        self.gaussians.densify_and_prune(min_opacity=self.gaussian_th, densify=False) 
        
        return self.data_update(self.h, self.w, self.current_window)

    def gaussian_reinit(self, rgbs, pointmaps, iteration_total=3000):
        self.reset()
        self.initialized = True
        rgbs = rgbs.cuda() / 255.0

        _, h1, w1, _ = pointmaps.shape
        rgbs = F.interpolate(rgbs, size=(h1, w1), mode='bilinear', align_corners=False)

        self.gaussians.extend_from_pcd_seq(rgb=rgbs[..., ::2, ::2], 
                                        pointmap=pointmaps[:, ::2, ::2], 
                                        normal=None, 
                                        conf=None, 
                                        submap_idx=0)
        
        self.gaussians.optimizer.zero_grad(set_to_none=True)   

        pbar = tqdm(range(iteration_total), desc="Gaussian Re-Training")
        for iteration in range(iteration_total):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_idx = viewpoint_idx_stack.pop(random.randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint = self.viewpoints[viewpoint_idx]
                    
            render_pkg = render(viewpoint, self.gaussians, self.background)

            image, depth = render_pkg["render"], render_pkg["depth"]
            image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"])
            
            gt_image = viewpoint.original_image.cuda()
            gt_depth = viewpoint.depth.to(dtype=torch.float32, device=image.device)[None]
            rgb_loss = 0.8 * torch.abs((gt_image - image)).mean() + 0.2 * (1.0 - ssim(image, gt_image))

            depth_pixel_mask = torch.logical_and(gt_depth > 0.001, depth > 0.001).view(*depth.shape)
            depth_loss = torch.abs(1./depth[depth_pixel_mask] - 1./gt_depth[depth_pixel_mask]).mean()
            # diff = torch.log(depth[depth_pixel_mask]) - torch.log(gt_depth[depth_pixel_mask])
            # depth_loss = (diff**2).mean() - diff.mean()**2

            # rendered_normal = render_pkg["normal"]
            depth_normal, _ = depth_to_normal(viewpoint, depth, world_frame=False)
            depth_normal = depth_normal.permute(2, 0, 1)
            # normal_loss = (1 - (rendered_normal * depth_normal).sum(dim=0)).mean()

            gt_depth_normal, _ = depth_to_normal(viewpoint, gt_depth.detach(), world_frame=False)
            gt_depth_normal = gt_depth_normal.permute(2, 0, 1)
            gt_normal_loss = (1 - (depth_normal * gt_depth_normal).sum(dim=0)).mean()

            loss = rgb_loss + self.lambda_depth * depth_loss + self.lambda_normal * gt_normal_loss
      
            loss.backward()

            with torch.no_grad():  
                if iteration > 1000:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if (iteration + 1) % self.gaussian_update_every == 0:
                        self.gaussians.densify_and_prune(
                                self.opt_params.densify_grad_threshold,
                                self.gaussian_th,
                                self.gaussian_extent,
                                self.size_threshold
                            )
                    # if (iteration + 1) == iteration_total // 2:
                    #     self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                if self.verbose and (iteration % 2000 == 0):
                    for view in list(self.viewpoints.values()):
                        os.makedirs(f"{self.output_dir}/rendering_GRT_{iteration_total}", exist_ok = True)
                        self.viz(view, f"{self.output_dir}/rendering_GRT_{iteration_total}", iteration)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update()
    
    def global_BA(self, iteration_total, densify=True):
        '''jointly optimize both gaussians and camera poses: Global BA'''
        opt_params = []
        for idx, view in enumerate(self.viewpoints.values()):
            opt_params.append({
                    "params": [view.cam_rot_delta],
                    "lr": self.config["opt_params"]["pose_lr"]*2,
                    "name": "rot_{}".format(view.uid)})
            opt_params.append({
                    "params": [view.cam_trans_delta],
                    "lr": self.config["opt_params"]["pose_lr"]*10,
                    "name": "trans_{}".format(view.uid)})
            
            if self.config["Training"]["compensate_exposure"]:
                opt_params.append({
                        "params": [view.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_a_{}".format(view.uid)})
                opt_params.append({
                        "params": [view.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_b_{}".format(view.uid)})
            
        self.keyframe_optimizers = torch.optim.Adam(opt_params)

        self.keyframe_optimizers.zero_grad(set_to_none=True)
        self.gaussians.optimizer.zero_grad(set_to_none=True)   

        pbar = tqdm(range(iteration_total), desc="Global BA")
        for iteration in range(iteration_total):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_idx = viewpoint_idx_stack.pop(random.randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint = self.viewpoints[viewpoint_idx]
                    
            render_pkg = render(viewpoint, self.gaussians, self.background)

            image, depth = render_pkg["render"], render_pkg["depth"]
            image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"])
            
            image = (image.permute(1, 2, 0) @ viewpoint.exposure_a + viewpoint.exposure_b).permute(2, 0, 1)

            gt_image = viewpoint.original_image.cuda()
            gt_depth = viewpoint.depth.to(dtype=torch.float32, device=image.device)[None]
            rgb_loss = 0.8 * torch.abs((gt_image - image)).mean() + 0.2 * (1.0 - ssim(image, gt_image))

            depth_pixel_mask = torch.logical_and(gt_depth > 0.001, depth > 0.001).view(*depth.shape)
            depth_loss = torch.abs(1./depth[depth_pixel_mask] - 1./gt_depth[depth_pixel_mask]).mean()
            # diff = torch.log(depth[depth_pixel_mask]) - torch.log(gt_depth[depth_pixel_mask])
            # depth_loss = (diff**2).mean() - diff.mean()**2

            rendered_normal = render_pkg["normal"]
            depth_normal, _ = depth_to_normal(viewpoint, depth, world_frame=False)
            depth_normal = depth_normal.permute(2, 0, 1)
            normal_loss = (1 - (rendered_normal * depth_normal).sum(dim=0)).mean()

            gt_depth_normal, _ = depth_to_normal(viewpoint, gt_depth.detach(), world_frame=False)
            gt_depth_normal = gt_depth_normal.permute(2, 0, 1)
            gt_normal_loss = (1 - (depth_normal * gt_depth_normal).sum(dim=0)).mean()

            loss = rgb_loss + self.lambda_depth / 10 * depth_loss + self.lambda_normal * normal_loss + self.lambda_normal * gt_normal_loss

            loss.backward()

            with torch.no_grad():  
                if iteration < (iteration_total // 2) and iteration > 500 and densify:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if (iteration + 1) % self.gaussian_update_every  == 0:
                        self.gaussians.densify_and_prune(
                                self.opt_params.densify_grad_threshold,
                                self.gaussian_th,
                                self.gaussian_extent,
                                self.size_threshold
                            )

                    if (iteration + 1) % self.gaussian_reset == 0:
                        self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.keyframe_optimizers.step()
                update_pose(viewpoint)

                self.keyframe_optimizers.zero_grad(set_to_none=True)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if densify:
                    lr = self.gaussians.update_learning_rate(iteration)

                if self.verbose and iteration % 2000 == 0:
                    for view in list(self.viewpoints.values()):
                        os.makedirs(f"{self.output_dir}/rendering_GBA_{iteration_total}", exist_ok = True)
                        self.viz(view, f"{self.output_dir}/rendering_GBA_{iteration_total}", iteration)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update()

        Log("Map refinement done")


    def finalize(self):
        for idx, view in enumerate(self.viewpoints.items()):
            self.pose_refine([idx], self.fx, self.fy, self.cx, self.cy, iters=50, lr_ratio=2, return_args=False, alpha_th=0.0)
            
        self.global_BA(iteration_total=self.gaussians.max_steps)

        for idx, view in enumerate(self.viewpoints.items()):
            self.pose_refine([idx], self.fx, self.fy, self.cx, self.cy, iters=50, lr_ratio=2, return_args=False, alpha_th=0.0)
        # self.global_BA(iteration_total=5000, densify=False)

        self.pcd_viz(self.gaussians.max_steps)
        os.makedirs(f"{self.output_dir}/ckpt", exist_ok=True)
        torch.save((self.gaussians.capture(), self.gaussians.max_steps), f"{self.output_dir}/ckpt/gausaian_ckpt_{self.gaussians.max_steps}.pth")  # save gaussian model

        poses_c2w = []
        for view in self.viewpoints.values():
            T_w2c = torch.eye(4)
            T_w2c[:3, :3] = view.R.cpu()
            T_w2c[:3, 3] = view.T.cpu()
            T_c2w = torch.inverse(T_w2c)
            quat = torch.from_numpy(Rotation.from_matrix(T_c2w[:3, :3]).as_quat()).to(T_c2w.device)
            trans = T_c2w[:3, 3]
            poses_c2w.append(torch.cat([trans, quat], dim=-1))
       
        return torch.stack(poses_c2w)
    
    def save(self, idx):
        self.pcd_viz(idx)
        os.makedirs(f"{self.output_dir}/ckpt", exist_ok=True)
        torch.save((self.gaussians.capture(), idx), f"{self.output_dir}/ckpt/gausaian_ckpt_{idx}.pth")  # save gaussian model

    def load(self):
        (model_params, first_iter) = torch.load(f"{self.output_dir}/ckpt/gausaian_ckpt_{0}.pth")
        self.gaussians.restore(model_params, self.opt_params)

    @torch.no_grad()
    def eval_rendering(self, gtimages, gtdepthdir, traj, kf_idx, eval_all=False):
        if eval_all:
            eval_rendering(gtimages, gtdepthdir, traj, self.gaussians, self.output_dir, self.background,
                self.projection_matrix, self.K, kf_idx, iteration="after_opt", kf_views=self.viewpoints)
        eval_rendering_kf(self.viewpoints, self.gaussians, self.output_dir, self.background, iteration="after_opt")
