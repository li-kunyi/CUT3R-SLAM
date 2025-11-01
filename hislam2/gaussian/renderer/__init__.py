#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian.scene.gaussian_model import GaussianModel
import torch.nn.functional as F
from gaussian.utils.slam_utils import get_pose

# def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0):
#     """
#     Render the scene.

#     Background tensor (bg_color) must be on GPU!
#     """

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         projmatrix_raw=viewpoint_camera.projection_matrix,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     cov3D_precomp = None
#     scales = pc.get_scaling
#     opacity = pc.get_opacity
#     rotations = pc.get_rotation

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = pc.get_features
#     colors_precomp = None

#     rendered_image, radii, rendered_expected_depth, n_touched = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp,
#         theta = viewpoint_camera.cam_rot_delta,
#         rho = viewpoint_camera.cam_trans_delta,
#     )

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "depth": rendered_expected_depth,
#             "viewspace_points": means2D,
#             "visibility_filter" : radii > 0,
#             "radii": radii,
#             "n_touched": n_touched}


def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0, require_coord : bool = True, require_depth : bool = True):
    rel_w2c = get_pose(viewpoint_camera)

    transformed_xyz, transformed_rot = transform_to_frame(
        pc.get_xyz, pc.get_rotation, rel_w2c
    )
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    viewmatrix = torch.eye(4, device=transformed_xyz.device)
    projmatrix = (viewmatrix.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))).squeeze(0)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = 0.0,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_coord = require_coord,
        require_depth = require_depth,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = transformed_xyz
    means2D = screenspace_points
    cov3D_precomp = None
    scales = pc.get_scaling
    opacity = pc.get_opacity
    rotations = transformed_rot
    shs = pc.get_features
    colors_precomp = None

    rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,
            "expected_coord": rendered_expected_coord,
            "median_coord": rendered_median_coord,
            "depth": rendered_expected_depth,
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            "n_touched": None
            }

def transform_to_frame(gaussian_xyz, gaussian_rot, rel_w2c):
    pts_ones = torch.ones(gaussian_xyz.shape[0], 1, device=gaussian_xyz.device)
    pts4 = torch.cat((gaussian_xyz, pts_ones), dim=1)  # [N, 4]
    transformed_xyz = (rel_w2c @ pts4.T).T[:, :3]  # [N, 3]

    rotmat = rel_w2c[:3, :3]  # [3, 3]
    
    cam_rot = rotmat_to_quat(rotmat)  # [4]
    norm_gaussian_rot = F.normalize(gaussian_rot, dim=-1)
    norm_cam_rot = F.normalize(cam_rot, dim=-1)
    transformed_rot = quat_mult(norm_cam_rot, norm_gaussian_rot)  # [N, 4]

    return transformed_xyz, transformed_rot

def rotmat_to_quat(R):
    trace = R.trace()
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return torch.tensor([w, x, y, z], device=R.device)

def quat_mult(q1, q2):
    # Hamilton product of two quaternions
    # q1, q2: [4] or [N, 4]
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)