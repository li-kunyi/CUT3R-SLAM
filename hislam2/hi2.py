import os
import torch
import torch.nn.functional as F
import numpy as np
from src.dust3r.model import ARCroco3DStereo
from keyframe import KeyFrame
from motion_filter import MotionFilter
from track_frontend import TrackFrontend
from track_backend import TrackBackend
from util.trajectory_filler import PoseTrajectoryFiller
from factor_graph import FactorGraph
# from gs_backend import GSBackEnd
from gs_backend_per_frame import GSBackEnd
from util.utils import viz_pcd


class Hi2:
    def __init__(self, args, config):
        super(Hi2, self).__init__()
        device = "cuda"
        self.model = ARCroco3DStereo.from_pretrained('./checkpoints/cut3r_512_dpt_4_64.pth').to(device)
        self.model.eval()

        self.config = config
        self.args = args
        self.verbose = False
        self.output_dir = args.output
        self.images = {}
        
        self.use_gt = False
        self.downsample_ratio = 2

        # buffer, store images, depth, poses, intrinsics with enough optical flow
        self.keyframes = KeyFrame(config, args.image_size, args.buffer, self.downsample_ratio)
        self.graph = FactorGraph(self.keyframes, max_factors=48)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.model, self.keyframes, config["Tracking"]["motion_filter"])

        # frontend process
        self.tracker = TrackFrontend(self, self.keyframes, config["Tracking"]["frontend"], device='cuda:0')

        # backend process: loop closure
        self.backend = TrackBackend(self, self.keyframes, config["Tracking"]["frontend"], device='cuda:0')

        # mapping and bundle adjustment
        self.mapper = GSBackEnd(self, config, self.args.output)
        self.gs_iter_num = config["Mapping"]["itr_num"]

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self)    

        self.freeze_counter = 0

    def call_gs(self, viz_idx, submap_idx, iterations, intrinsics):
        tstamp = self.keyframes.tstamp[viz_idx].to(device='cpu')
        poses = self.keyframes.pose[viz_idx]

        images = []
        for t in tstamp:
            images.append(self.images[t.item()].to(device='cpu'))
        images = torch.cat(images, dim=0)

        pointmaps = self.keyframes.submap_ds[submap_idx]
        confs = self.keyframes.conf_ds[submap_idx]
        depths = self.keyframes.depth[viz_idx]

        data = {'viz_idx':  viz_idx,
                'submap_idx': submap_idx,
                'tstamp':   tstamp,
                'poses':    poses,
                'images':   images,
                'pointmaps':  pointmaps[:len(viz_idx)],
                'confs':    confs[:len(viz_idx)],
                'normals':  None,
                'depths':   depths,
                'intrinsics':  intrinsics}
        
        data['pose_updates'] = None
        
        updated_data, updated_idx = self.mapper.run(data, iterations)

        self.keyframes.pose[updated_idx] = updated_data['poses'].float()
        depth = updated_data['depths'].cpu()
        mask = depth > 0
        self.keyframes.depth[updated_idx][mask] = depth[mask]
        updated_pointmap = updated_data['pointmaps'].cpu()
        
        self.keyframes.submap_ds[np.array(updated_idx) // 5, np.array(updated_idx) % 5] = updated_pointmap[:, ::self.downsample_ratio, ::self.downsample_ratio]
        self.keyframes.submap_ds[:submap_idx + 1, -1] = self.keyframes.submap_ds[1:submap_idx + 2, 0]

        if self.verbose:
            t1 = viz_idx[-1]
            pts = (self.keyframes.submap_ds[:submap_idx + 1, :-1].flatten(0, 1))[:t1-1].cpu().numpy()
            confs = (self.keyframes.conf_ds[:submap_idx + 1, :-1].flatten(0, 1))[:t1-1].cpu().numpy()
            images = self.keyframes.image[:t1-1].permute(0, 2, 3, 1).cpu().numpy() / 255.0
            images = images[:, ::self.downsample_ratio, ::self.downsample_ratio]
            viz_pcd(pts, images, os.path.join(self.output_dir, "after_gs"), tstamp[-1], conf=confs, th=0.0)

    def run(self, tstamp, image, intrinsics, image_ds, intrinsics_ds, second_last_frame=False, last_frame=False):        
        with torch.no_grad():
            self.images[tstamp] = image

            # keyframe selection based on similarity
            self.filterx.kfFilter(tstamp, image_ds, intrinsics=intrinsics_ds, second_last_frame=second_last_frame, last_frame=last_frame)

            # frontend: tracking
            run_backend, viz_idx, submap_idx = self.tracker.run(tstamp, last_frame=last_frame)

            # backend: point-based loop closure
            lc_did = False
            if run_backend and not last_frame:
                if self.freeze_counter > 0:
                    lc_did, updates = self.backend.run()
                    if lc_did:
                        self.freeze_counter = 0
                else:
                    self.freeze_counter += 1

        if lc_did:            
            updated_data, updated_idx = self.mapper.gaussain_update(updates)

            self.keyframes.pose[updated_idx] = updated_data['poses'].float()

            updated_pointmap = updated_data['pointmaps'].cpu()
            self.keyframes.submap_ds[np.array(updated_idx) // 5, np.array(updated_idx) % 5] = updated_pointmap[:, ::self.downsample_ratio, ::self.downsample_ratio]
            self.keyframes.submap_ds[:submap_idx + 1, -1] = self.keyframes.submap_ds[1:submap_idx + 2, 0]

        # backend: mapping and local BA
        if viz_idx is not None:
            self.call_gs(viz_idx, submap_idx, self.gs_iter_num, intrinsics.squeeze())


    def test(self, tstamp, image, intrinsics, depth, pose, second_last_frame=False, last_frame=False):        
        with torch.no_grad():
            self.images[tstamp] = image

            # keyframe selection based on similarity
            self.filterx.kfFilter(tstamp, image, intrinsics=intrinsics, depth=depth, pose=pose, second_last_frame=second_last_frame, last_frame=last_frame)

            # frontend: tracking
            run_backend, viz_idx, submap_idx = self.tracker.test(tstamp, last_frame=last_frame)

        # backend: mapping and local BA
        if viz_idx is not None:
            self.call_gs(viz_idx, submap_idx, self.gs_iter_num, intrinsics.squeeze())


    def terminate(self, tstamp, fill=False, eval_render=False, gaussian_retrain=False, add_kf=False):
        """ terminate the visualization process, return poses [t, q] """
        last_kf_idx = self.keyframes.counter.value
        if gaussian_retrain:
            # Gaussian re-training
            pointmaps = self.keyframes.submap_ds[:(last_kf_idx - 1)//5, :-1].flatten(0, 1)
            if (last_kf_idx - 1) % 5 != 0:
                pointmaps = torch.cat([pointmaps, self.keyframes.submap_ds[(last_kf_idx - 1)//5, :((last_kf_idx - 1)%5)]], dim=0)
            N_p = pointmaps.shape[0]

            tstamp = self.keyframes.tstamp[:(last_kf_idx-1)].to(device='cpu')
            images = []
            for t in tstamp:
                images.append(self.images[t.item()].to(device='cpu'))
            images = torch.cat(images, dim=0)

            N_i = images.shape[0]

            N = min(N_p, N_i)
            images = images[:N]
            pointmaps = pointmaps[:N]
            self.mapper.gaussian_reinit(images, pointmaps, iteration_total=10000)

        # to better finalize the gaussian model, add new kf if the interval is too large
        if add_kf:
            kf_tstamp = self.keyframes.tstamp[:(last_kf_idx - 1)]
            for i, tstamp in enumerate(kf_tstamp[:-1]):
                if kf_tstamp[i+1] - tstamp > 30:
                    kf_img = self.keyframes.image[i]
                    kf_pose = self.keyframes.pose[i]
                    kf_pointmap = self.keyframes.submap_ds[i//5, i%5]
                    kf_depth = self.keyframes.depth[i]
                    kf_sub_idx = i//5
                    H, W = kf_img.shape[-2:]

                    if (kf_tstamp[i+1] - tstamp) > 60:
                        N = 2
                        interval = (kf_tstamp[i+1] - tstamp) // 3
                    else:
                        N = 1
                        interval = (kf_tstamp[i+1] - tstamp) // 2

                    for j in range(N):
                        new_kf_tstamp = int((tstamp + interval * (j + 1)).cpu().item())
                        new_img = self.images[new_kf_tstamp]
                        _new_img = F.interpolate(new_img, size=(H, W), mode='bilinear', align_corners=False)[0]

                        new_pose, new_depth, new_pointmap, new_conf = self.tracker.predict(_new_img, 
                                                                                           kf_img, 
                                                                                           kf_pose, 
                                                                                           kf_depth, 
                                                                                           kf_pointmap)
                        self.mapper.add_new_view(new_img, 
                                                 new_pose[None], 
                                                 new_depth[None], 
                                                 new_pointmap[None], 
                                                 new_conf[None], 
                                                 new_kf_tstamp, 
                                                 kf_sub_idx)

        updated_poses = self.mapper.finalize()
        updated_poses = updated_poses[:last_kf_idx-1]
        self.keyframes.pose[:last_kf_idx-1] = torch.tensor(updated_poses)

        traj_full = self.keyframes.pose
        if fill:
            # estimate pose of every frame
            traj_full = self.traj_filler.run(self.images)  # c2w [trans, quat]

        if eval_render:
            self.mapper.eval_rendering(self.images, self.args.gtdepthdir, traj_full, self.keyframes.tstamp[:self.keyframes.counter.value].to(device='cpu'),
                                       eval_all=fill)
        
        return traj_full.numpy()
