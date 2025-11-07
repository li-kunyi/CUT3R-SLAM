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

from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l1_loss_weight(network_output, gt):
    image = gt.detach().cpu().numpy().transpose((1, 2, 0))
    rgb_raw_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    sobelx = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_merge = np.sqrt(sobelx * sobelx + sobely * sobely) + 1e-10
    sobel_merge = np.exp(sobel_merge)
    sobel_merge /= np.max(sobel_merge)
    sobel_merge = torch.from_numpy(sobel_merge)[None, ...].to(gt.device)

    return torch.abs((network_output - gt) * sobel_merge).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def masked_ssim(img1, img2, mask, win_size=11, k1=0.01, k2=0.03, L=1.0, valid_ratio_thresh=0.8):
    # img1, img2: [B, C, H, W], in [0,1]
    # mask: [B,1,H,W], 1 for valid pixels, 0 for invalid
    B, C, H, W = img1.shape
    padding = win_size // 2
    # kernel of ones for local sums
    kernel = torch.ones((1, 1, win_size, win_size), device=img1.device)

    # compute local count of valid pixels per window
    valid_count = F.conv2d(mask, kernel, padding=padding)  # [B,1,H,W]
    # threshold for considering a window "valid"
    win_area = win_size * win_size
    min_valid = valid_ratio_thresh * win_area

    # compute local sums but weighted by mask (zero out invalid pixels)
    img1_masked = img1 * mask
    img2_masked = img2 * mask
    # per-channel conv: do conv per channel by reshaping
    def local_sum(x):
        # x: [B, C, H, W]; convolve each channel with ones
        x_resh = x.view(B*C, 1, H, W)
        s = F.conv2d(x_resh, kernel, padding=padding)
        return s.view(B, C, H, W)

    sum1 = local_sum(img1_masked)
    sum2 = local_sum(img2_masked)
    sum1_sq = local_sum(img1_masked * img1_masked)
    sum2_sq = local_sum(img2_masked * img2_masked)
    sum12 = local_sum(img1_masked * img2_masked)

    # avoid division by zero: where valid_count==0 set to 1 temporarily
    vc = valid_count.clamp(min=1.0)

    mu1 = sum1 / vc
    mu2 = sum2 / vc
    sigma1_sq = sum1_sq / vc - mu1 * mu1
    sigma2_sq = sum2_sq / vc - mu2 * mu2
    sigma12 = sum12 / vc - mu1 * mu2

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # SSIM map per channel
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2))

    # average over channels
    ssim_map = ssim_map.mean(dim=1, keepdim=True)  # [B,1,H,W]

    # mask windows that don't have enough valid pixels
    window_valid_mask = (valid_count >= min_valid).float()  # [B,1,H,W]

    # final SSIM: average ssim_map * window_valid_mask normalized by number of valid windows
    valid_windows = window_valid_mask.sum(dim=[1,2,3]).clamp(min=1.0)  # [B]
    ssim_per_batch = (ssim_map * window_valid_mask).sum(dim=[1,2,3]) / valid_windows  # [B]

    # return 1 - ssim (loss), or ssim_per_batch as similarity
    return ssim_per_batch.mean()


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
