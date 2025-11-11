import cv2
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F

from torchvision import transforms
# from midas.omnidata import OmnidataModel
from util.utils import compute_patch_overlap_ratio


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, model, keyframes, config, device="cuda:0"):
        self.model = model
        self.keyframes = keyframes
        self.thresh = config["thresh"]
        self.skip = config["skip"]
        self.init_thresh = config["init_thresh"] if "init_thresh" in config else self.thresh
        self.device = device
        self.kf_every = config["kf_every"]

        self.count = 0
        self.omni_dep = None

        self.skip_blur = config["skip_blur"]
        self.cache = [None]*5
        self.shapeness = [0]*5

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def prior_extractor(self, im_tensor):
        input_size = im_tensor.shape[-2:]
        trans_totensor = transforms.Compose([transforms.Resize((512, 512), antialias=True)])
        im_tensor = trans_totensor(im_tensor).cuda()
        if self.omni_dep is None:
            self.omni_dep = OmnidataModel('depth', 'pretrained_models/omnidata_dpt_depth_v2.ckpt', device="cuda:0")
            self.omni_normal = OmnidataModel('normal', 'pretrained_models/omnidata_dpt_normal_v2.ckpt', device="cuda:0")
        depth = self.omni_dep(im_tensor)[None] * 50
        depth = F.interpolate(depth, input_size, mode='bicubic')
        depth = depth.float().squeeze()
        normal = self.omni_normal(im_tensor) * 2.0 - 1.0
        normal = F.interpolate(normal, input_size, mode='bicubic')
        normal = normal.float().squeeze()
        return depth, normal.permute(1, 2, 0)
    

    @torch.amp.autocast('cuda', enabled=True)
    @torch.no_grad()
    def kfFilter(self, tstamp, image, intrinsics=None, pose=None, depth=None, second_last_frame=False, last_frame=False):
        """ main update operation - run on every frame in keyframes """
        # Evaluate the quality of keyframes frames to help decide whether to skip blurry frames
        # s = sharpness(image[0].permute(1,2,0).cpu().numpy())

        # normalize images
        inputs = image[None].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        ### always add first frame to the depth keyframes ###
        # self.keyframes.counter.value will increase once self.keyframes.append is called
        if self.kf_every > 0:
            compute_overlap = False
        else:
            compute_overlap = True
        if self.keyframes.counter.value == 0 or last_frame or second_last_frame:
            # depth, normal = self.prior_extractor(inputs[0])
            normal = None
            view1 = {}
            view1['img'] = self.model.normalize(image[0]).unsqueeze(0).to('cuda')  # normalize the image to [-1, 1]

            ##### Encode frames
            feat1, pos1, _ = self.model.encode_image(view1)
            feat1 = feat1.squeeze(0)
            self.keyframes.append(tstamp, image[0], pose, 1.0, depth, normal, intrinsics, feat1, pos1)
        else:      
            if compute_overlap and tstamp % self.skip == 0:
                kf_idx = self.keyframes.counter.value
                feat0 = self.keyframes.featI[kf_idx - 1]

                view1 = {}
                view1['img'] = self.model.normalize(image[0]).unsqueeze(0).to('cuda')  # normalize the image to [-1, 1]

                ##### Encode frames
                feat1, pos1, _ = self.model.encode_image(view1)
                feat1 = feat1.squeeze(0)
                overlap_ratio = compute_patch_overlap_ratio(feat0, feat1)   
            elif not compute_overlap and tstamp % self.kf_every == 0:
                kf_idx = self.keyframes.counter.value
                feat0 = self.keyframes.featI[kf_idx - 1]

                view1 = {}
                view1['img'] = self.model.normalize(image[0]).unsqueeze(0).to('cuda')  # normalize the image to [-1, 1]

                ##### Encode frames
                feat1, pos1, _ = self.model.encode_image(view1)
                feat1 = feat1.squeeze(0)
            else:
                overlap_ratio = 1.0
                feat1 = None
                pos1 = None
                
            if (compute_overlap and overlap_ratio < self.thresh) or (not compute_overlap and tstamp % self.kf_every == 0):
                # index_min = np.argmax(self.shapeness)
                # check if the current frame is blurry, if so, use the previous frame(most sharp one)
                # if self.skip_blur and self.shapeness[index_min] > s:
                #     tstamp, image, pose, depth, intrinsics, inputs, feat1, pos1 = self.cache[index_min]
                # self.shapeness = [0]*5
                # self.cache = [None]*5

                # depth, normal = self.prior_extractor(inputs[0])
                normal = None
                # self.count = 0
                self.keyframes.append(tstamp, image[0], pose, None, depth, normal, intrinsics, feat1, pos1)
            # else:
            #     self.shapeness[tstamp%5] = s
            #     self.cache[tstamp%5] = [tstamp, image, pose, depth, intrinsics, inputs, feat1, pos1]
            #     self.count += 1
