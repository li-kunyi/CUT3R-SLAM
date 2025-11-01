import yaml
import numpy as np
import rich
import copy
import torch
from matplotlib import cm
import os
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
import torchvision
import math

_log_styles = {
    "GSBackend": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "PGBA": "bold blue",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="GSBackend"):
    style = get_style(tag)
    rich.print(f"\n[{style}]{tag}:[/{style}]", *args)


def load_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # merge per dataset cfg. and main cfg.
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively. dict1 get masked by dict2, and we retuen dict1.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def colorize_np(x, cmap_name='jet', range=None):
    if range is not None:
        vmin, vmax = range
    else:
        vmin, vmax = np.percentile(x, (1, 99))

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]
    return x_new


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj

def se3_exp_map(se3):
    """
    Exponential map from se(3) to SE(3)
    Input: se3 [6] (axis-angle + translation)
    Output: SE(3) matrix [4, 4]
    """
    omega = se3[:3]
    v = se3[3:]

    theta = torch.norm(omega)
    if theta < 1e-5:
        R = torch.eye(3).to(se3.device)
        V = torch.eye(3).to(se3.device)
    else:
        axis = omega / theta
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=se3.device)
        R = torch.eye(3).to(se3.device) + \
            torch.sin(theta) * K + \
            (1 - torch.cos(theta)) * K @ K
        V = torch.eye(3).to(se3.device) + \
            (1 - torch.cos(theta)) / theta * K + \
            (theta - torch.sin(theta)) / (theta ** 2) * K @ K

    t = V @ v.view(3, 1)
    T = torch.eye(4).to(se3.device)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def viz_pcd(pts, images=None, pcd_dir=None, idx=0, name=None, conf=None, th=0):
    # Save point cloud
    if conf is not None:
        pts = pts[conf>th]
        if images is not None:
            images = images[conf>th]
    if name is None:
        name = f"frame_{idx}_conf{th}.ply"

    os.makedirs(pcd_dir, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3))
    if images is not None:
        pcd.colors = o3d.utility.Vector3dVector(images.reshape(-1, 3))
    o3d.io.write_point_cloud(os.path.join(pcd_dir, name), pcd)


def viz_depth(W, H, pts, w2c, output_dir):
    pts = pts.reshape(-1, 3)
    pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))  # Convert to homogeneous coordinates
    pts_camera_view = (w2c @ pts_homogeneous.T).T[:, :3]  # Transform to current camera view
            
    # Extract depth (z-coordinate in the camera view)
    depth_map = pts_camera_view[:, 2].reshape(H, W)
        
    # Normalize depth for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            
    # Save the colored depth map
    cv2.imwrite(output_dir, depth_colored)


def viz_map(map, output_dir, colorize=True):
    # Normalize depth for visualization
    map_normalized = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX)
    
    map_out = map_normalized.astype(np.uint8)
    if colorize:
        map_out = cv2.applyColorMap(map_out, cv2.COLORMAP_JET)
    else:
        map_out = cv2.cvtColor(map_out, cv2.COLOR_RGB2BGR)
    # Save the colored depth map
    cv2.imwrite(output_dir, map_out)


def compute_distance(pointmap_i, pointmap_j, T_world_to_cam_j, fx, fy, cx, cy, device="cuda"):
    "project point_i to view j"
    H, W, _ = pointmap_i.shape
    N = H * W

    # [N, 3]
    points_i = pointmap_i.reshape(-1, 3)
    ones = torch.ones((N, 1), device=points_i.device)
    points_i_h = torch.cat([points_i, ones], dim=-1)  # [N, 4]

    # Transform into camera j frame
    points_cam_j = (T_world_to_cam_j @ points_i_h.T).T[:, :3]  # [N, 3]
    x, y, z = points_cam_j[:, 0], points_cam_j[:, 1], points_cam_j[:, 2]

    # Create mask for z > 0 (i.e., in front of the camera)
    valid_z_mask = (z > 0)

    # Filter valid 3D points
    x, y, z = x[valid_z_mask], y[valid_z_mask], z[valid_z_mask]
    points_i_valid = points_i[valid_z_mask]

    # Project to normalized image coordinates [-1, 1]
    u = fx * x / z + cx
    v = fy * y / z + cy
    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).view(1, -1, 1, 2)  # [1, N_valid, 1, 2]

    # Sample pointmap_j (shape: [1, 3, H, W])
    pointmap_j_tensor = pointmap_j.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(
        pointmap_j_tensor, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
    ).squeeze().T  # [N_valid, 3]

    # Optional: mask zero-sampled points (outside image bounds)
    mask_valid_sample = (sampled.abs().sum(dim=1) > 0)
    points_i_final = points_i_valid[mask_valid_sample]
    points_j_final = sampled[mask_valid_sample]

    # Compute L2 reprojection error
    dist = (points_i_final - points_j_final).norm(dim=1)
    return dist.mean()

def total_variance(image):
    grad_x = image[:, :, :-1] - image[:, :, 1:]
    grad_y = image[:, :-1, :] - image[:, 1:, :]
    
    grad_x = torch.cat((grad_x, grad_x[:, :, -1:]), dim=2)
    grad_y = torch.cat((grad_y, grad_y[:, -1:, :]), dim=1)
    return grad_x, grad_y

def TV_loss(depth, normal=None, image=None, conf_masks=None):
    depth_grad_x, depth_grad_y = total_variance(depth)

    if image is not None:
        gray_image = 0.2989 * image[:, :, :, 0] + 0.5870 * image[:, :, :, 1] + 0.1140 * image[:, :, :, 2]
        gt_img_grad_x, gt_img_grad_y = total_variance(gray_image)

        img_grad_magnitude = torch.sqrt(gt_img_grad_x**2 + gt_img_grad_y**2)
        weights = torch.exp(-img_grad_magnitude*5)
    else:
        weights = torch.ones_like(depth_grad_x)

    if conf_masks is None:
        conf_masks = torch.ones_like(depth_grad_x)

    tv_loss_x = torch.mean((torch.abs(depth_grad_x) * weights * conf_masks))
    tv_loss_y = torch.mean((torch.abs(depth_grad_y) * weights * conf_masks))
    loss = tv_loss_x + tv_loss_y

    if normal is not None:
        normal_grad_x, normal_grad_y = total_variance(normal)
        tv_normal_loss_x = torch.mean(torch.abs(normal_grad_x).mean(dim=-1) * weights * conf_masks)
        tv_normal_loss_y = torch.mean(torch.abs(normal_grad_y).mean(dim=-1) * weights * conf_masks)
        tv_normal_loss = tv_normal_loss_x + tv_normal_loss_y
        loss += 0.05 * tv_normal_loss
        
    return loss, weights

def get_depth_normal(pointmap, T_world_to_cam):
    """
    pointmap_i: [B, H, W, 3]
    pointmap_j: [B, H, W, 3]
    T_world_to_cam_j: [B, 4, 4]
    fx, fy, cx, cy: float or tensor of shape [B]
    """
    B, H, W, _ = pointmap.shape
    N = H * W

    device = pointmap.device

    # [B, N, 3]
    points_i = pointmap.view(B, -1, 3)
    ones = torch.ones((B, N, 1), device=device)
    points_i_h = torch.cat([points_i, ones], dim=-1).float()  # [B, N, 4]

    # Transform into camera j frame
    points_cam_j = torch.bmm(points_i_h, T_world_to_cam.transpose(1, 2))[:, :, :3]  # [B, N, 3]
    x, y, z = points_cam_j[..., 0], points_cam_j[..., 1], points_cam_j[..., 2]

    depth = z.reshape(B, H, W)
    invalid_mask = (depth < 0) | torch.isinf(depth) | torch.isnan(depth)
    depth = torch.where(invalid_mask, torch.zeros_like(depth), depth)

    # get normal
    normal = torch.zeros_like(pointmap)
    dx = torch.cat([pointmap[:, 2:, 1:-1] - pointmap[:, :-2, 1:-1]], dim=0)
    dy = torch.cat([pointmap[:, 1:-1, 2:] - pointmap[:, 1:-1, :-2]], dim=1)
    depth_grad = torch.cross(dx, dy, dim=-1)
    _normal = torch.nn.functional.normalize(depth_grad, dim=-1)
    normal[:, 1:-1, 1:-1, :] = _normal

    return depth, normal

def compute_distance_batchwise(pointmaps, images, depths, normals, T_w2c, index_i, index_j, 
                               fx, fy, cx, cy, img_weights=None, conf_masks=None, z_buffer_th=0.5, tstamp=0):
    """
    pointmap_i: [B, H, W, 3]
    pointmap_j: [B, H, W, 3]
    T_world_to_cam_j: [B, 4, 4]
    fx, fy, cx, cy: float or tensor of shape [B]
    """
    pointmap_i = pointmaps[index_i]  # [B, H, W, 3]
    image_i = images[index_i]  # [B, H, W, 3]
    normal_i = normals[index_i]  # [B, H, W, 3]

    pointmap_j = pointmaps[index_j]  # [B, H, W, 3]
    image_j = images[index_j]  # [B, H, W, 3]
    normal_j = normals[index_j]  # [B, H, W, 3]
    T_world_to_cam_j = T_w2c[index_j]  # [B, 4, 4]
    depth_j = depths[index_j]  # [B, H, W]

    B, H, W, _ = pointmap_i.shape
    N = H * W

    device = pointmap_i.device

    # [B, N, 3]
    points_i = pointmap_i.view(B, -1, 3)
    ones = torch.ones((B, N, 1), device=device)
    points_i_h = torch.cat([points_i, ones], dim=-1)  # [B, N, 4]
        
    # Transform into camera j frame
    T = T_world_to_cam_j  # [B, 4, 4]
    points_cam_j = torch.bmm(points_i_h, T.transpose(1, 2))[:, :, :3]  # [B, N, 3]
    x, y, z = points_cam_j[..., 0], points_cam_j[..., 1], points_cam_j[..., 2]
    # valid mask for z > 0
    valid_mask = (z > 0)
    # Project
    z = z.clamp(min=1e-8)  # Avoid division by zero
    u = fx * x / z + cx
    v = fy * y / z + cy
    # Normalize to [-1, 1] for grid_sample
    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).view(B, N, 1, 2)  # [B, N, 1, 2]

    concat_tensor = torch.cat([
        pointmap_j.permute(0, 3, 1, 2),        # [B, 3, H, W]
        depth_j.unsqueeze(1),                   # [B, 1, H, W]
        normal_j.permute(0, 3, 1, 2),           # [B, 3, H, W]
        image_j.permute(0, 3, 1, 2)             # [B, 3, H, W]
    ], dim=1)  # [B, 10, H, W]

    sampled_all = F.grid_sample(
        concat_tensor, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
    )  # [B, 10, N, 1]
    sampled_all = sampled_all.squeeze(-1).permute(0, 2, 1)  # [B, N, 10]

    sampled = sampled_all[:, :, 0:3]   # [B, N, 3]
    depth_sampled = sampled_all[:, :, 3]     # [B, N]
    normal_sampled = sampled_all[:, :, 4:7]   # [B, N, 3]
    img_sampled = sampled_all[:, :, 7:10]  # [B, N, 3]

    # Get valid sampled points
    mask_valid_sample = (sampled.abs().sum(dim=2) > 0) & valid_mask  # [B, N]
    points_i_valid = torch.where(mask_valid_sample.unsqueeze(-1), points_i, torch.zeros_like(points_i))
    points_j_valid = torch.where(mask_valid_sample.unsqueeze(-1), sampled, torch.zeros_like(sampled))
    # Get valid sampled normals
    normal_i_valid = torch.where(mask_valid_sample.unsqueeze(-1), normal_i.view(B, -1, 3), torch.zeros_like(normal_i.view(B, -1, 3)))
    normal_j_valid = torch.where(mask_valid_sample.unsqueeze(-1), normal_sampled, torch.zeros_like(normal_sampled))
    # Get valid sampled images
    img_i_valid = torch.where(mask_valid_sample.unsqueeze(-1), image_i.view(B, -1, 3), torch.zeros_like(image_i.view(B, -1, 3)))
    img_j_valid = torch.where(mask_valid_sample.unsqueeze(-1), img_sampled, torch.zeros_like(img_sampled))

    # Compute loss
    if img_weights is not None:
        weight_i = img_weights[index_i] * img_weights[index_j]
        img_i_valid = img_i_valid * weight_i.view(B, -1, 1)
        img_j_valid = img_j_valid * weight_i.view(B, -1, 1)
        points_i_valid = points_i_valid * weight_i.view(B, -1, 1)
        points_j_valid = points_j_valid * weight_i.view(B, -1, 1)

    if conf_masks is not None:
        mask_valid_sample = mask_valid_sample & conf_masks[index_i].view(B, -1)

    # Test occlusion
    z_mask = torch.abs(z - depth_sampled) < z_buffer_th
    z_mask = mask_valid_sample & z_mask  # [B, N]
    photo_mean_dist = compute_loss(img_i_valid, img_j_valid, z_mask, loss_type='huber')
    normal_mean_dist = compute_loss(normal_i_valid, normal_j_valid, z_mask, loss_type='l1')
    mean_dist = compute_loss(points_i_valid, points_j_valid, z_mask, loss_type='huber', delta=0.1)

    return mean_dist, normal_mean_dist, photo_mean_dist


def compute_distance_v2(pointmaps, images, depths, T_w2c, index_i, index_j, 
                      fx, fy, cx, cy, z_buffer_th=0.5):
    """
    pointmap_i: [B, H, W, 3]
    pointmap_j: [B, H, W, 3]
    T_world_to_cam_j: [B, 4, 4]
    fx, fy, cx, cy: float or tensor of shape [B]
    """
    pointmap_i = pointmaps[index_i]  # [B, H, W, 3]
    image_i = images[index_i]  # [B, H, W, 3]

    pointmap_j = pointmaps[index_j]  # [B, H, W, 3]
    image_j = images[index_j]  # [B, H, W, 3]
    T_world_to_cam_j = T_w2c[index_j]  # [B, 4, 4]
    depth_j = depths[index_j]  # [B, H, W]

    B, H, W, _ = pointmap_i.shape
    N = H * W

    device = pointmap_i.device

    # [B, N, 3]
    points_i = pointmap_i.view(B, -1, 3)
    ones = torch.ones((B, N, 1), device=device)
    points_i_h = torch.cat([points_i, ones], dim=-1)  # [B, N, 4]
        
    # Transform into camera j frame
    T = T_world_to_cam_j  # [B, 4, 4]
    points_cam_j = torch.bmm(points_i_h, T.transpose(1, 2))[:, :, :3]  # [B, N, 3]
    x, y, z = points_cam_j[..., 0], points_cam_j[..., 1], points_cam_j[..., 2]
    # valid mask for z > 0
    valid_mask = (z > 0)
    # Project
    z = z.clamp(min=1e-8)  # Avoid division by zero
    u = fx * x / z + cx
    v = fy * y / z + cy
    # Normalize to [-1, 1] for grid_sample
    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).view(B, N, 1, 2)  # [B, N, 1, 2]

    concat_tensor = torch.cat([
        pointmap_j.permute(0, 3, 1, 2),        # [B, 3, H, W]
        depth_j.unsqueeze(1),                   # [B, 1, H, W]
        image_j.permute(0, 3, 1, 2)             # [B, 3, H, W]
    ], dim=1)  # [B, 10, H, W]

    sampled_all = F.grid_sample(
        concat_tensor, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
    )  # [B, 10, N, 1]
    sampled_all = sampled_all.squeeze(-1).permute(0, 2, 1)  # [B, N, 10]

    sampled = sampled_all[:, :, 0:3]   # [B, N, 3]
    depth_sampled = sampled_all[:, :, 3]     # [B, N]
    img_sampled = sampled_all[:, :, 4:]  # [B, N, 3]

    # Get valid sampled points
    mask_valid_sample = (sampled.abs().sum(dim=2) > 0) & valid_mask  # [B, N]
    points_i_valid = torch.where(mask_valid_sample.unsqueeze(-1), points_i, torch.zeros_like(points_i))
    points_j_valid = torch.where(mask_valid_sample.unsqueeze(-1), sampled, torch.zeros_like(sampled))
    # Get valid sampled images
    img_i_valid = torch.where(mask_valid_sample.unsqueeze(-1), image_i.view(B, -1, 3), torch.zeros_like(image_i.view(B, -1, 3)))
    img_j_valid = torch.where(mask_valid_sample.unsqueeze(-1), img_sampled, torch.zeros_like(img_sampled))

    # Test occlusion
    z_mask = torch.abs(z - depth_sampled) < z_buffer_th
    z_mask = mask_valid_sample & z_mask  # [B, N]
    photo_mean_dist = compute_loss(img_i_valid, img_j_valid, z_mask, loss_type='l1')
    mean_dist = compute_loss(points_i_valid, points_j_valid, z_mask, loss_type='l1')

    return mean_dist, photo_mean_dist

def depth_to_pointmap(depth, T_c2w=None, fx=None, fy=None, cx=None, cy=None, world_coord=True):
    """
    depth: [B, H, W]
    T_c2w: [B, 4, 4]
    fx, fy, cx, cy: float or tensor
    return: [B, H, W, 3]
    """
    B, H, W = depth.shape
    device = depth.device

    # meshgrid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    x = (grid_x.float() - cx) / fx
    y = (grid_y.float() - cy) / fy
    z = depth

    # [B, H, W, 3] camera coordinates
    pts_cam = torch.stack([x.expand(B, -1, -1) * z, y.expand(B, -1, -1) * z, z], dim=-1)  # [B, H, W, 3]
    pts_cam_h = torch.cat([pts_cam, torch.ones_like(z)[..., None]], dim=-1)  # [B, H, W, 4]
    pts_cam_h = pts_cam_h.view(B, -1, 4)  # [B, H*W, 4]

    if world_coord:
        # world coordinates
        pts_world = torch.bmm(pts_cam_h, T_c2w.transpose(1, 2))  # [B, H*W, 4]
        pts_world = pts_world[..., :3].view(B, H, W, 3)
        return pts_world
    else:
        pts_cam = pts_cam[..., :3].view(B, H, W, 3)
        return pts_cam

def compute_distance_from_depth(images, depths, T_w2c, index_i, index_j, 
                               fx, fy, cx, cy, z_buffer_th=0.1):
    """
    pointmap_i: [B, H, W, 3]
    pointmap_j: [B, H, W, 3]
    T_world_to_cam_j: [B, 4, 4]
    fx, fy, cx, cy: float or tensor of shape [B]
    """
    # project depth to world
    T_c2w = torch.inverse(T_w2c)
    pointmaps = depth_to_pointmap(depths, T_c2w, fx, fy, cx, cy)

    image_i = images[index_i]  # [B, H, W, 3]
    depth_i = depths[index_i]  # [B, H, W]
    pointmap_i = pointmaps[index_i]

    image_j = images[index_j]  # [B, H, W, 3]
    depth_j = depths[index_j]  # [B, H, W]
    pointmap_j = pointmaps[index_j]
    T_w2c_j = T_w2c[index_j]  # [B, 4, 4]

    B, H, W = depth_i.shape
    N = H * W
    device = depth_i.device

    # [B, N, 3]
    points_i = pointmap_i.view(B, -1, 3)
    ones = torch.ones((B, N, 1), device=device)
    points_i_h = torch.cat([points_i, ones], dim=-1)  # [B, N, 4]
        
    # Transform into camera j frame
    T = T_w2c_j  # [B, 4, 4]
    points_cam_j = torch.bmm(points_i_h, T.transpose(1, 2))[:, :, :3]  # [B, N, 3]
    x, y, z = points_cam_j[..., 0], points_cam_j[..., 1], points_cam_j[..., 2]
    # valid mask for z > 0
    valid_mask = (z > 0)
    # Project
    z = z.clamp(min=1e-8)  # Avoid division by zero
    u = fx * x / z + cx
    v = fy * y / z + cy
    # Normalize to [-1, 1] for grid_sample
    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).view(B, N, 1, 2)  # [B, N, 1, 2]

    concat_tensor = torch.cat([
        pointmap_j.permute(0, 3, 1, 2),        # [B, 3, H, W]
        depth_j.unsqueeze(1),                   # [B, 1, H, W]
        image_j.permute(0, 3, 1, 2)             # [B, 3, H, W]
    ], dim=1)  # [B, 10, H, W]

    sampled_all = F.grid_sample(
        concat_tensor, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
    )  # [B, 10, N, 1]
    sampled_all = sampled_all.squeeze(-1).permute(0, 2, 1)  # [B, N, 10]

    sampled = sampled_all[:, :, 0:3]   # [B, N, 3]
    depth_sampled = sampled_all[:, :, 3]     # [B, N]
    img_sampled = sampled_all[:, :, 4:]  # [B, N, 3]

    # Get valid sampled points
    mask_valid_sample = (sampled.abs().sum(dim=2) > 0) & valid_mask  # [B, N]
    points_i_valid = torch.where(mask_valid_sample.unsqueeze(-1), points_i, torch.zeros_like(points_i))
    points_j_valid = torch.where(mask_valid_sample.unsqueeze(-1), sampled, torch.zeros_like(sampled))
    # Get valid sampled images
    img_i_valid = torch.where(mask_valid_sample.unsqueeze(-1), image_i.view(B, -1, 3), torch.zeros_like(image_i.view(B, -1, 3)))
    img_j_valid = torch.where(mask_valid_sample.unsqueeze(-1), img_sampled, torch.zeros_like(img_sampled))

    # Test occlusion
    z_mask = torch.abs(z - depth_sampled) < z_buffer_th
    z_mask = mask_valid_sample & z_mask  # [B, N]
    photo_mean_dist = compute_loss(img_i_valid, img_j_valid, z_mask, loss_type='huber')
    mean_dist = compute_loss(points_i_valid, points_j_valid, z_mask, loss_type='huber', delta=0.1)

    return mean_dist, photo_mean_dist


def compute_reproject_error(pointmaps, T_w2c, fx, fy, cx, cy):
    B, H, W, _ = pointmaps.shape
    N = H * W

    device = pointmaps.device

    # [B, N, 3]
    points_i = pointmaps.view(B, -1, 3)
    ones = torch.ones((B, N, 1), device=device)
    points_i_h = torch.cat([points_i, ones], dim=-1)  # [B, N, 4]

    ####### Transform into camera i(self) frame #######
    points_cam_i = torch.bmm(points_i_h, T_w2c.transpose(1, 2))[:, :, :3]  # [B, N, 3]
    x, y, z = points_cam_i[..., 0], points_cam_i[..., 1], points_cam_i[..., 2]
    
    # Project
    z = z.clamp(min=1e-8)  # Avoid division by zero
    u = fx * x / z + cx
    v = fy * y / z + cy
    # Normalize to [-1, 1] for grid_sample
    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1

    # GT image coords
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    u_gt = (grid_x.reshape(1, -1).expand(B, -1).float() / (W - 1)) * 2 - 1
    v_gt = (grid_y.reshape(1, -1).expand(B, -1).float() / (W - 1)) * 2 - 1

    # PnP reprojection constraints
    # Only compute reprojection error where the difference is less than or equal to 10 pixels
    diff_u = torch.abs(u_norm - u_gt)
    diff_v = torch.abs(v_norm - v_gt)
    mask = (diff_u <= 5.0) & (diff_v <= 5.0)
    reproj_dist = ((diff_u + diff_v) * mask)

    return reproj_dist.mean()


def compute_loss(pred, target, mask_valid_sample, loss_type='l2', delta=0.1):
    diff = pred - target  # [B, N, 3]

    if loss_type == 'huber':
        abs_diff = torch.abs(diff)
        quadratic = torch.minimum(abs_diff, torch.tensor(delta))
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic**2 + delta * linear
        return (loss.sum(dim=2) * mask_valid_sample).sum() / (mask_valid_sample.sum() + 1e-8)
    
    elif loss_type == 'l2':
        dist = torch.norm(diff, dim=2)  # [B, N]
    elif loss_type == 'l1':
        dist = torch.abs(diff).sum(dim=2)  # [B, N]

    # Mean over valid
    num_valid = mask_valid_sample.sum(dim=1)
    # if (num_valid < 10).any():
    #     print("Warning: Too few valid points when reprojection, this might not be expected.") 
        
    num_valid = num_valid.clamp(min=1)
    sum_dist = (dist * mask_valid_sample).sum(dim=1)  # [B]
    mean_dist = (sum_dist / num_valid).mean()  # [B]

    return mean_dist


def pose_smoothness_loss(T_w2c):
    """
    Compute smoothness loss between consecutive poses in SE(3)
    Args:
        T_w2c: [B, 4, 4] tensor of pose matrices (world-to-camera)
    Returns:
        smooth_loss: scalar tensor
    """
    B = T_w2c.shape[0]
    loss = 0.0
    for i in range(B - 1):
        T1 = T_w2c[i]     # [4, 4]
        T2 = T_w2c[i + 1] # [4, 4]

        # Compute relative transform: T_rel = T2 @ inv(T1)
        T_rel = T2 @ torch.inverse(T1)  # [4, 4]

        # Extract translation and rotation
        trans = T_rel[:3, 3]                        # [3]
        rot = T_rel[:3, :3]                         # [3, 3]

        # Rotation loss: use rotation matrix to angle (theta)
        cos_theta = (torch.trace(rot) - 1) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical stability
        angle = torch.acos(cos_theta)  # radians

        # Translation norm
        trans_norm = torch.norm(trans)

        loss = loss + angle ** 2 + trans_norm ** 2

    return loss / (B - 1)

def quaternion_to_rotation_matrix(quat):
    """Convert normalized quaternion to rotation matrix. quat: [B, 4]"""
    # Normalize
    quat = quat / quat.norm(p=2, dim=-1, keepdim=True)
    x, y, z, w = quat.unbind(-1)
    B = quat.shape[0]

    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w,
        2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y
    ], dim=-1).reshape(B, 3, 3)
    return R

def pose_vec_to_matrix(pose):
    """pose: [B, 7] → transform: [B, 4, 4]"""
    B = pose.shape[0]
    trans = pose[:, :3]
    quat = pose[:, 3:]
    rot = quaternion_to_rotation_matrix(quat)
    # rot = quat2rotation(quat)
    T = torch.eye(4, device=pose.device).repeat(B, 1, 1)
    T[:, :3, :3] = rot
    T[:, :3, 3] = trans
    return T


def if_bundle_adjustment(pose0, pose1, trans_thresh=0.2, rot_thresh=5.0):
    """
    pose0, pose1: torch.Tensor of shape (7,), [t (3,), q (4,)]
    trans_thresh: translation threshold in meters
    rot_thresh: rotation threshold in degrees
    """

    t0, q0 = pose0[:3], pose0[3:]
    t1, q1 = pose1[:3], pose1[3:]
    trans_drift = torch.norm(t0 - t1)

    q0 = F.normalize(q0, dim=0)
    q1 = F.normalize(q1, dim=0)

    dot = torch.abs(torch.sum(q0 * q1))
    dot = torch.clamp(dot, max=1.0)
    rot_drift_rad = 2 * torch.acos(dot)
    rot_drift_deg = torch.rad2deg(rot_drift_rad)

    if trans_drift > trans_thresh or rot_drift_deg > rot_thresh:
        return True
    return False

def compute_patch_overlap_ratio(feat0, feat1, threshold=0.7):
    feat0 = F.normalize(feat0[1:], dim=1)
    feat1 = F.normalize(feat1[1:], dim=1)

    sim = torch.mm(feat0, feat1.T)
    max_sim, _ = sim.max(dim=1)

    matched = (max_sim > threshold).float()
    ratio = matched.mean().item()

    return ratio

def umeyama_alignment(source, target):
    """
    source: Nx3, target: Nx3
    Returns: scale, rotation (3x3), translation (3,)
    """
    assert source.shape == target.shape
    N = source.shape[0]

    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)

    src_centered = source - mu_src
    tgt_centered = target - mu_tgt

    cov_matrix = tgt_centered.T @ src_centered / N
    U, S, Vt = np.linalg.svd(cov_matrix)

    D = np.diag([1, 1, np.linalg.det(U @ Vt)])
    R = U @ D @ Vt

    var_src = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = np.sum(S * D.diagonal()) / var_src

    t = mu_tgt - scale * R @ mu_src

    return scale, R, t


def voxel_downsample(points, voxel_size):
    if len(points) == 0:
        return points
    coords = torch.floor(points / voxel_size).int()
    _, unique_indices = torch.unique(coords, dim=0, return_inverse=False, return_counts=False, return_index=True)
    return points[unique_indices]


def check_view_frustum_overlap(current_pose, other_poses, hfov_deg=90.0, max_depth=5.0):
    c = current_pose[:3, 3]
    z = current_pose[:3, 2]

    centers = other_poses[:, :3, 3]
    vecs = centers - c[None]  # 向量 [N, 3]
    dists = torch.norm(vecs, dim=1)
    vecs_normed = vecs / (dists[:, None] + 1e-8)

    cos_theta = torch.sum(vecs_normed * z[None], dim=1)
    angle_thresh = torch.cos(torch.tensor(hfov_deg / 2 * math.pi / 180.0, device=z.device))

    mask = (cos_theta > angle_thresh) & (dists < max_depth)
    return mask

def compute_hfov_deg(fx, W):
    hfov_rad = 2 * torch.atan(W / (2 * fx))
    hfov_deg = hfov_rad * 180.0 / math.pi
    return hfov_deg



def compute_camera_distance(pose_current_lc, pointmap_current_lc,
                    pose_matched_lc, pointmap_matched_lc,
                    fx, fy, cx, cy, H, W,
                    dist_th=0.5, reproj_th=5.0):
    """
    Return True if pose distance is small or reprojection error is small.
    Uses grid_sample to interpolate matched pointmap.
    """
    device = pose_current_lc.device

    # Get 4x4 poses
    T_curr = pose_vec_to_matrix(pose_current_lc)
    T_match = pose_vec_to_matrix(pose_matched_lc).double()

    # Camera center distance
    cam_dist = torch.norm(T_curr[:, :3, 3] - T_match[:, :3, 3])
    if cam_dist < dist_th:
        return True
    
    ##TODO to test
    # [H, W, 3] -> [H*W, 3]
    B, H, W, _ = pointmap_current_lc.shape
    pts = pointmap_current_lc.view(B, -1, 3)
    ones = torch.ones((B, H*W, 1), device=device)
    points_i_h = torch.cat([pts, ones], dim=-1)  # [B, N, 4]

    # Transform current points to matched camera frame
    pts_trans = torch.bmm(points_i_h, T_match.transpose(1, 2))[:, :, :3]  # [B, N, 3]
    x, y, z = pts_trans[..., 0], pts_trans[..., 1], pts_trans[..., 2]
    # valid mask for z > 0
    valid_mask = (z > 0)
    # Project
    z = z.clamp(min=1e-8)  # Avoid division by zero
    u = fx * x / z + cx
    v = fy * y / z + cy
    # Normalize to [-1, 1] for grid_sample
    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).view(B, H*W, 1, 2).float()  # [B, N, 1, 2]
    # Prepare matched pointmap: [1, 3, H, W]
    matched_map = pointmap_matched_lc.permute(0, 3, 1, 2).float()  # [1, 3, H, W]

    # Sample matched points
    sampled = F.grid_sample(matched_map, grid, mode='bilinear', align_corners=True)  # [1, 3, N, 1]
    sampled = sampled.squeeze(0).squeeze(-1).permute(1, 0)  # [N, 3]

    # Compute reprojection error
    reproj_error = (sampled - pts_trans)[valid_mask].norm(dim=-1)
    mean_err = reproj_error.mean()

    return mean_err < reproj_th

def gray_world_normalize(images):
    # images: [3, H, W]
    mean_rgb = images.mean(dim=(1, 2), keepdim=True)  # [3, 1, 1]
    target = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(3, 1, 1)
    normalized = images * (target / (mean_rgb + 1e-6))
    return normalized#.clamp(0.0, 1.0)

def compute_alignment_error(point_map1, conf1, point_map2, conf2, conf_threshold, s, R, t):
    """
    Compute the average point alignment error (using only original inputs)
    
    Args:
    point_map1: target point map (b, h, w, 3)
    conf1: target confidence map (b, h, w)
    point_map2: source point map (b, h, w, 3)
    conf2: source confidence map (b, h, w)
    conf_threshold: confidence threshold
    s, R, t: transformation parameters
    """
    b1, h1, w1, _ = point_map1.shape
    b2, h2, w2, _ = point_map2.shape
    b = min(b1, b2)
    h = min(h1, h2)
    w = min(w1, w2)
    
    target_points = []
    source_points = []
    
    for i in range(b):
        mask1 = conf1[i, :h, :w] > conf_threshold
        mask2 = conf2[i, :h, :w] > conf_threshold
        valid_mask = mask1 & mask2

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue
            
        t_pts = point_map1[i, :h, :w][idx]
        s_pts = point_map2[i, :h, :w][idx]
        
        target_points.append(t_pts)
        source_points.append(s_pts)
    
    if len(target_points) == 0:
        print("Warning: No matching point pairs found for error calculation")
        return np.nan
    
    all_target = np.concatenate(target_points, axis=0)
    all_source = np.concatenate(source_points, axis=0)
    
    transformed = (s * (R @ all_source.T)).T + t
    
    errors = np.linalg.norm(transformed - all_target, axis=1)
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    
    print(f"Alignment error statistics [using {len(errors)} points]: "
          f"mean={mean_error:.4f}, std={std_error:.4f}, "
          f"median={median_error:.4f}, max={max_error:.4f}")
    
    return mean_error

def weighted_estimate_sim3(source_points, target_points, weights):
    """
    source_points:  (Nx3)
    target_points:  (Nx3)
    :weights:  (N,) [0,1]
    """
    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")
    
    normalized_weights = weights / total_weight

    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt(np.sum(normalized_weights * np.sum(src_centered**2, axis=1)))
    scale_tgt = np.sqrt(np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1)))
    s = scale_tgt / scale_src

    weighted_src = (s * src_centered) * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]
    
    H = weighted_src.T @ weighted_tgt

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


def huber_loss(r, delta):
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta))

def robust_weighted_estimate_sim3(src, tgt, init_weights, delta=0.1, max_iters=20, tol=1e-9):
    """
    src:  (Nx3)
    tgt:  (Nx3)
    init_weights:  (N,)
    """
    s, R, t = weighted_estimate_sim3(src, tgt, init_weights)
    prev_error = float('inf')
    
    for iter in range(max_iters):

        transformed = s * (src @ R.T) + t
        residuals = np.linalg.norm(tgt - transformed, axis=1)  # (N,)
        # print(f'Residuals: {np.mean(residuals)}')
        
        abs_res = np.abs(residuals)
        huber_weights = np.ones_like(residuals)
        large_res_mask = abs_res > delta
        huber_weights[large_res_mask] = delta / abs_res[large_res_mask]
        
        combined_weights = init_weights * huber_weights
        
        combined_weights /= (np.sum(combined_weights) + 1e-12)
        
        s_new, R_new, t_new = weighted_estimate_sim3(src, tgt, combined_weights)

        param_change = np.abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1)/2)))
        current_error = np.sum(huber_loss(residuals, delta) * init_weights)
        
        if (param_change < tol and rot_angle < np.radians(0.1)) or \
           (abs(prev_error - current_error) < tol * prev_error):
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error
    
    return s, R, t


def weighted_align_point_maps(point_map1, conf1, point_map2, conf2, conf_threshold):
    """ point_map2 -> point_map1"""
    b1, _, _, _ = point_map1.shape
    b2, _, _, _ = point_map2.shape
    b = min(b1, b2)
    
    aligned_points1 = []
    aligned_points2 = []
    confidence_weights = []

    for i in range(b):
        mask1 = conf1[i] > conf_threshold
        mask2 = conf2[i] > conf_threshold
        valid_mask = mask1 & mask2

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue

        pts1 = point_map1[i][idx]
        pts2 = point_map2[i][idx]

        combined_conf = np.sqrt(conf1[i][idx] * conf2[i][idx])
        
        aligned_points1.append(pts1)
        aligned_points2.append(pts2)
        confidence_weights.append(combined_conf)

    if len(aligned_points1) == 0:
        raise ValueError("No matching point pairs were found!")

    all_pts1 = np.concatenate(aligned_points1, axis=0)
    all_pts2 = np.concatenate(aligned_points2, axis=0)
    all_weights = np.concatenate(confidence_weights, axis=0)

    # print(f"The number of corresponding points matched: {all_pts1.shape[0]}")
    
    s, R, t = robust_weighted_estimate_sim3(all_pts2, 
                                            all_pts1, 
                                            all_weights,
                                            delta=0.1,
                                            max_iters=5,
                                            tol=1e-9
                                            )

    # mean_error = compute_alignment_error(
    #     point_map1, conf1, 
    #     point_map2, conf2, 
    #     conf_threshold, 
    #     s, R, t
    # )
    # print(f'Mean error: {mean_error}')

    return s, R, t


def sobel_edges(x):
    # x: [3, H, W]
    x = x.unsqueeze(0)  # -> [1, 3, H, W]
    
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(x, sobel_x.expand(3, -1, -1, -1), padding=1, groups=3)
    grad_y = F.conv2d(x, sobel_y.expand(3, -1, -1, -1), padding=1, groups=3)
    
    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return edges.squeeze(0)  # -> [3, H, W]

def gaussian_blur(x, kernel_size=5, sigma=1.0):
    # x: [3, H, W]
    
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    kernel = kernel.expand(x.size(0), 1, -1, -1)
    return F.conv2d(x.unsqueeze(0), kernel, padding=kernel_size//2, groups=3).squeeze(0)