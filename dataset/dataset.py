import glob
import os
import csv
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from natsort import natsorted
from scipy.spatial.transform import Rotation

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def get_dataset(cfg, input_folder, scale=1.0):
    return dataset_dict[cfg['Cam']['dataset']](cfg, input_folder, scale)


class BaseDataset(Dataset):
    def __init__(self, cfg, input_folder, scale):
        super(BaseDataset, self).__init__()
        self.name = cfg['Cam']['dataset']
        self.scale = scale
        self.png_depth_scale = cfg['Cam']['png_depth_scale']
        self.distortion = np.array(cfg['Cam']['distortion']) if 'distortion' in cfg['Cam'] else None
        self.crop_size = cfg['Cam']['crop_size'] if 'crop_size' in cfg['Cam'] else None
        self.input_folder = input_folder
        self.crop_edge = cfg['Cam']['crop_edge']
        self.mode = cfg['mode'] if 'mode' in cfg else 'rgb'

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        # load rgb
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)

        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        # color_data = color_data.astype(np.float32) / 255.
        H, W, C = color_data.shape
        h1 = (H // 64) * 16
        w1 = (h1 * 4) // 3
        color_data = cv2.resize(color_data, (w1, h1))
        color_data = torch.from_numpy(color_data)

        # load depth
        depth_data = -1
        if self.mode == 'rgbd':
            depth_path = self.depth_paths[index]
            if '.png' in depth_path:
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                #depth_data = cv2.imdecode(np.fromfile(depth_path, dtype=np.uint16), -1)
            else:
                raise ValueError('No depth image found!')

        # crop the image
        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            color_data = color_data.permute(1, 2, 0).contiguous()

            if self.mode == 'rgbd':
                depth_data = F.interpolate(
                    depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
        
        # remove artifacts on the edge of the image    
        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid values on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            if self.mode == 'rgbd':
                depth_data = depth_data[edge:-edge, edge:-edge]
            
            # Adjust fx, fy, cx, cy after cropping
            self.cx -= edge
            self.cy -= edge
    
        pose = self.poses[index]
        pose[:3, 3] *= self.scale
        # Convert pose into quaternion and output pose as [T, q]
        T = pose[:3, 3].numpy()
        R = pose[:3, :3].numpy()
        q = Rotation.from_matrix(R).as_quat()  # Quaternion in [x, y, z, w] format
        pose_output = np.concatenate([T, q])  # Combine translation and quaternion

        self.intrinsic[[0,2]] *= (w1 / W)
        self.intrinsic[[1,3]] *= (h1 / H)
        
        return index, color_data.permute(2, 0, 1), depth_data, self.intrinsic, pose_output



class ScanNet(BaseDataset):
    def __init__(self, cfg, input_folder, scale):
        super(ScanNet, self).__init__(cfg, input_folder, scale)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['Cam']['H'], cfg['Cam'][
            'W'], cfg['Cam']['fx'], cfg['Cam']['fy'], cfg['Cam']['cx'], cfg['Cam']['cy']
        
        self.input_folder = os.path.join(self.input_folder)#, 'frames')

        self.color_paths = natsorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))

        self.depth_paths = natsorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))

        self.id_map = {}

        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)

        self.intrinsic = np.array([self.fx, self.fy, self.cx, self.cy])

    def load_poses(self, path):
        self.poses = []
        pose_paths = natsorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class Replica(BaseDataset):
    def __init__(self, cfg, args, scale):
        super(Replica, self).__init__(cfg, args, scale)

        self.H, self.W= cfg['Cam']['H'], cfg['Cam']['W']
        
        self.hfov = 90
        # the pin-hole camera has the same value for fx and fy
        self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        # self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
        self.fy = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0

        self.color_paths = natsorted(glob.glob(f'{self.input_folder}/rgb/rgb_*.png'))
        self.depth_paths = natsorted(glob.glob(f'{self.input_folder}/depth/depth_*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj_w_c.txt')
        
        self.intrinsic = np.array([self.fx, self.fy, self.cx, self.cy])

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, scale):
        super(TUM_RGBD, self).__init__(cfg, args, scale)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)
        self.semantic = False

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "tum_rgbd": TUM_RGBD,
}
