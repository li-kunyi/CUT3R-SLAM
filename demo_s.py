import os    # nopep8
import sys   # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), 'hislam2'))   # nopep8
import time
import torch
import cv2
import re
import os
import argparse
import numpy as np
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))
from natsort import natsorted
from tqdm import tqdm
from hislam2.hi2 import Hi2
from hislam2.util.utils import load_config

import json

def save_config(image, args, cfg):
    shape_info = f"image.shape: {tuple(image.shape)}\n"

    args_dict = vars(args)

    if not isinstance(cfg, dict):
        cfg_dict = dict(cfg)
    else:
        cfg_dict = cfg

    with open("image_shape.txt", "w") as f:
        f.write(shape_info)
        f.write("\nArgs:\n")
        f.write(json.dumps(args_dict, indent=4))
        f.write("\n\nCfg:\n")
        f.write(json.dumps(cfg_dict, indent=4))


def show_image(image, depth_prior, depth, normal):
    from hislam2.util.utils import colorize_np
    image = image[[2,1,0]].permute(1, 2, 0).cpu().numpy()
    depth = colorize_np(np.concatenate((depth_prior.cpu().numpy(), depth.cpu().numpy()), axis=1), range=(0, 4))
    normal = normal.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('rgb / prior normal / aligned prior depth / JDSA depth', np.concatenate((image / 255.0, (normal[...,[2,1,0]]+1.)/2., depth), axis=1)[::2,::2])
    cv2.waitKey(1)


def mono_stream(imagedir, calib, undistort=False, cropborder=False, start=0, length=100000):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    K = np.array([[calib[0], 0, calib[2]],[0, calib[1], calib[3]],[0,0,1]])

    image_list = natsorted(os.listdir(imagedir))[start:start+length]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        intrinsics = torch.tensor(calib[:4])
        intrinsics_ds = torch.tensor(calib[:4])
        if len(calib) > 4 and undistort:
            image = cv2.undistort(image, K, calib[4:])
        if cropborder > 0:
            image = image[cropborder:-cropborder, cropborder:-cropborder]
            intrinsics[2:] -= cropborder
            intrinsics_ds[2:] -= cropborder

        # for tracking
        h0, w0, _ = image.shape
        h1 = int((512 / w0 * h0) // 16) * 16
        w1 = 512
        image_ds = cv2.resize(image, (w1, h1))
        image_ds = torch.as_tensor(image_ds).permute(2, 0, 1)

        intrinsics_ds[0] *= (w1 / w0)
        intrinsics_ds[1] *= (h1 / h0)
        intrinsics_ds[2] *= (w1 / w0)
        intrinsics_ds[3] *= (h1 / h0)

        # for gaussian mapping
        h2 = int(512 / w0 * h0)
        w2 = 512
        image = cv2.resize(image, (w2, h2))
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics[0] *= (w2 / w0)
        intrinsics[1] *= (h2 / h0)
        intrinsics[2] *= (w2 / w0)
        intrinsics[3] *= (h2 / h0)

        is_last = (t == len(image_list)-1)
        yield (t, image[None], intrinsics[None], image_ds[None], intrinsics_ds[None], is_last)

    time.sleep(10)


def save_trajectory(hi2, traj_full, imagedir, output, start=0):
    t = hi2.keyframes.counter.value - 1
    tstamps = hi2.keyframes.tstamp[:t]
    poses_kf = hi2.keyframes.pose[:t]  # c2w
    np.save("{}/intrinsics.npy".format(output), hi2.keyframes.intrinsic[0].cpu().numpy())

    tstamps_full = np.array([float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1]) for x in natsorted(os.listdir(imagedir))[start:]])[..., np.newaxis]
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    ttraj_kf = np.concatenate([tstamps_kf, poses_kf.cpu().numpy()], axis=1)
    # np.savetxt(f"{output}/traj_kf.txt", ttraj_kf)                     #  for evo evaluation 
    np.savetxt(f"{output}/traj_kf.txt", 
               ttraj_kf, 
               fmt="%.4f %.7f %.7f %.7f %.7f %.7f %.7f %.7f")

    if traj_full is not None:
        ttraj_full = np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1)  # c2w
        np.savetxt(f"{output}/traj_full.txt", ttraj_full)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--posedir", type=str, default=None, help="path to pose directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--config", type=str, help="path to configuration file")
    parser.add_argument("--output", default='outputs/demo', help="path to save output")
    parser.add_argument("--gtdepthdir", type=str, default=None, help="optional for evaluation, assumes 16-bit depth scaled by 6553.5")

    parser.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "pretrained_models/droid.pth"))
    parser.add_argument("--buffer", type=int, default=-1, help="number of keyframes to buffer (default: 1/10 of total frames)")
    parser.add_argument("--undistort", action="store_true", help="undistort images if calib file contains distortion parameters")
    parser.add_argument("--cropborder", type=int, default=0, help="crop images to remove black border")

    parser.add_argument("--droidvis", action="store_true")
    parser.add_argument("--gsvis", action="store_true")

    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--length", type=int, default=100000, help="number of frames to process")
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/spann3r.pth', help='ckpt path')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    cfg = load_config(args.config)    

    N = len(os.listdir(args.imagedir))
    args.buffer = min(1000, N // 10 + 150) if args.buffer < 0 else args.buffer
    pbar = tqdm(range(N), desc="Processing keyframes", initial=args.start)

    hi2 = None
    for t, image, intrinsics, image_ds, intrinsics_ds, is_last in mono_stream(args.imagedir, args.calib, args.undistort, args.cropborder, args.start, args.length):
        if hi2 is None:
            args.image_size = [image_ds.shape[2], image_ds.shape[3]]
            hi2 = Hi2(args,cfg)

            save_config(image, args, cfg)

        second_last_frame = (t + args.start == N - 2)
        last_frame = (t + args.start == N - 1)
        hi2.run(t, image, intrinsics, image_ds, intrinsics_ds, second_last_frame=second_last_frame, last_frame=last_frame)

        pbar.set_description(f"Processing {hi2.keyframes.counter.value - 1} th keyframe")
        pbar.update()

    pbar.close()

    # save gaussian model
    hi2.mapper.save(0)
    # offline refinement(gloabal BA)
    # traj = None
    traj = hi2.terminate(t+1, fill=False, eval_render=True, gaussian_retrain=False, add_kf=True)
    
    # save pose(c2w) for evo_ape
    save_trajectory(hi2, traj, args.imagedir, args.output, start=args.start)

    print("Done")

    # os.system(f'evo_ape tum /home/kunyi/work/data/tum/rgbd_dataset_freiburg1_360/groundtruth.txt {args.output}/traj_kf.txt -vas --save_results {args.output}/evo.zip --save_plot {args.output}/evo.png  --no_warnings > {args.output}/ape.txt')

    # evo_ape tum data/ScanNet/scene0000_00/traj.txt outputs/scene0000/traj_kf.txt -vas --no_warnings > outputs/scene0000/ape.txt
    # workstation:
    # evo_ape tum data/ScanNet/scene0000_00/traj.txt outputs/scene0000_250703_00/traj_kf.txt -vas --save_plot outputs/scene0000_250703_00/ape_plot.png --no_warnings > outputs/scene0000_250703_00/ape.txt