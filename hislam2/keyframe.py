import torch
from torch.multiprocessing import Value

class KeyFrame:
    def __init__(self, config, image_size, buffer, downsample_ratio):
        '''
            store keyframes
        '''
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]
        self.is_initialized = False
        self.config = config
        self.downsample_ratio = downsample_ratio

        ### state attributes
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float)#.share_memory_()
        self.image = torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.uint8)

        # camera parameters
        self.intrinsic = torch.zeros(buffer, 4, device="cpu", dtype=torch.float)#.share_memory_()
        self.pose = torch.zeros(buffer, 7, device="cpu", dtype=torch.float)#.share_memory_()
        self.pose[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cpu")  # initialize poses to identity transformation

        # point map and confidence
        self.submap_ds = torch.ones(buffer//5, 6, ht // self.downsample_ratio, wd // self.downsample_ratio, 3, device="cpu", dtype=torch.float)#.share_memory_()
        self.conf_ds = torch.zeros(buffer//5, 6, ht // self.downsample_ratio, wd // self.downsample_ratio, device="cpu", dtype=torch.float)#.share_memory_()

        # depth and normal
        # self.normal = torch.ones(buffer, ht, wd, 3, device="cpu", dtype=torch.float)#.share_memory_()
        self.depth = torch.ones(buffer, ht, wd, device="cpu", dtype=torch.float)#.share_memory_()

        ### feature attributes
        self.featI = torch.zeros(buffer, (ht//16) * (wd//16), 1024, dtype=torch.float, device="cuda")#.share_memory_()
        self.pos = torch.zeros(buffer, (ht//16) * (wd//16), 2, dtype=torch.int64, device="cuda")#.share_memory_()


    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.image[index] = item[1]

        if item[2] is not None:
            self.pose[index] = item[2]

        if item[3] is not None:
            pass

        if item[4] is not None:
            self.depth[index] = item[4]
            # pass

        if item[5] is not None:
            # self.normal_priors[index] = item[5]
            pass

        if item[6] is not None:
            self.intrinsic[index] = item[6]
        else:
            self.intrinsic[index] = self.intrinsic[0].clone()

        if len(item) > 7:
            self.featI[index] = item[7]

        if len(item) > 8:
            self.pos[index] = item[8]


    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.image[index],
                self.pose[index],
                # self.submap[index],
                # self.conf[index],
                # self.depth_priors[index],
                # self.normal_priors[index],
                # self.dscale[index],
                # self.doffset[index]
                )

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    def shift(self, ix, n=1):
        with self.get_lock():
            self.tstamp[ix+n:self.counter.value+n] = self.tstamp[ix:self.counter.value].clone()
            self.image[ix+n:self.counter.value+n] = self.image[ix:self.counter.value].clone()
            self.dirty[ix+n:self.counter.value+n] = self.dirty[ix:self.counter.value].clone()
            self.pose[ix+n:self.counter.value+n] = self.pose[ix:self.counter.value].clone()
            self.disps[ix+n:self.counter.value+n] = self.disps[ix:self.counter.value].clone()
            self.disps_prior[ix+n:self.counter.value+n] = self.disps_prior[ix:self.counter.value].clone()
            self.disps_up[ix+n:self.counter.value+n] = self.disps_up[ix:self.counter.value].clone()
            self.disps_prior_up[ix+n:self.counter.value+n] = self.disps_prior_up[ix:self.counter.value].clone()
            self.intrinsic[ix+n:self.counter.value+n] = self.intrinsic[ix:self.counter.value].clone()
            self.counter.value += n

    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean().item() * self.config['Dataset']['scale_multiplier']
            self.pose[:self.counter.value,:3] *= s
            self.disps[:self.counter.value] /= s
            self.disps_up[:self.counter.value] /= s
            self.dscales[:self.counter.value] /= s
            self.dirty[:self.counter.value] = True

        

