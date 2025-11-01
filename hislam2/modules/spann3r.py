import copy
import torch
import numpy as np
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
import torchvision.transforms as tvf
from croco.models.blocks import Block
from dust3r.model import AsymmetricCroCo3DStereo


class SpatialMemory():
    def __init__(self, model, mem_dropout=None, 
                 long_mem_size=4000, work_mem_size=5, 
                 attn_thresh=5e-4, sim_thresh=0.95, 
                 save_attn=False, num_patches=None):
        self.norm_q = model.norm_q
        self.norm_k = model.norm_k
        self.norm_v = model.norm_v
        self.mem_dropout = mem_dropout
        self.attn_thresh = attn_thresh
        self.long_mem_size = long_mem_size
        self.work_mem_size = work_mem_size
        self.top_k = long_mem_size
        self.save_attn = save_attn
        self.sim_thresh = sim_thresh
        self.num_patches = num_patches
        self.init_mem()
    
    def init_mem(self):
        self.local_mem_k = None
        self.local_mem_v = None
        self.mem_k = None
        self.mem_v = None
        self.mem_c = None
        self.mem_count = None
        self.mem_attn = None
        self.mem_pts = None
        self.mem_imgs = None
        self.lm = 0
        self.wm = 0
        if self.save_attn:
            self.attn_vis = None

    def add_mem_k(self, feat):
        if self.mem_k is None:
            self.mem_k = feat
        else:
            self.mem_k = torch.cat((self.mem_k, feat), dim=1)

        return self.mem_k
    
    def add_mem_v(self, feat):
        if self.mem_v is None:
            self.mem_v = feat
        else:
            self.mem_v = torch.cat((self.mem_v, feat), dim=1)

        return self.mem_v

    def add_mem_c(self, feat):
        if self.mem_c is None:
            self.mem_c = feat
        else:
            self.mem_c = torch.cat((self.mem_c, feat), dim=1)

        return self.mem_c
    
    def add_mem_pts(self, pts_cur):
        if pts_cur is not None:
            if self.mem_pts is None:
                self.mem_pts = pts_cur
            else:
                self.mem_pts = torch.cat((self.mem_pts, pts_cur), dim=1)
    
    def add_mem_img(self, img_cur):
        if img_cur is not None:
            if self.mem_imgs is None:
                self.mem_imgs = img_cur
            else:
                self.mem_imgs = torch.cat((self.mem_imgs, img_cur), dim=1)

    def add_mem(self, feat_k, feat_v, pts_cur=None, img_cur=None):  
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]
            
        if self.mem_count is None:
            self.mem_count = torch.zeros_like(feat_k[:, :, :1])
            self.mem_attn = torch.zeros_like(feat_k[:, :, :1])
        else:
            self.mem_count += 1
            self.mem_count = torch.cat((self.mem_count, torch.zeros_like(feat_k[:, :, :1])), dim=1)
            self.mem_attn = torch.cat((self.mem_attn, torch.zeros_like(feat_k[:, :, :1])), dim=1)
        
        self.add_mem_k(feat_k)
        self.add_mem_v(feat_v)
        self.add_mem_pts(pts_cur)
        self.add_mem_img(img_cur)
    
    def check_sim(self, feat_k, thresh=0.7):
        # Do correlation with working memory
        if self.mem_k is None or thresh==1.0:
            return False
        
        wmem_size = self.wm * self.num_patches

        # wm: BS, T, 196, C
        wm = self.mem_k[:, -wmem_size:].reshape(self.mem_k.shape[0], -1, self.num_patches, self.mem_k.shape[-1])

        feat_k_norm = F.normalize(feat_k, p=2, dim=-1)
        wm_norm = F.normalize(wm, p=2, dim=-1)

        corr = torch.einsum('bpc,btpc->btp', feat_k_norm, wm_norm)

        mean_corr = torch.mean(corr, dim=-1)

        if mean_corr.max() > thresh:
            print('Similarity detected:', mean_corr.max())
            return True
    
        return False

    def add_mem_check(self, feat_k, feat_v, pts_cur=None, img_cur=None, prune=True):
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]

        if self.check_sim(feat_k, thresh=self.sim_thresh):
            return
        
        self.add_mem(feat_k, feat_v, pts_cur, img_cur)
        self.wm += 1

        if self.wm > self.work_mem_size:
            self.wm -= 1
            if self.long_mem_size == 0:
                self.mem_k = self.mem_k[:, self.num_patches:]
                self.mem_v = self.mem_v[:, self.num_patches:]
                self.mem_count = self.mem_count[:, self.num_patches:]
                self.mem_attn = self.mem_attn[:, self.num_patches:]
                print('Memory pruned:', self.mem_k.shape)
            else:
                self.lm += self.num_patches
        
        # long + short -> long
        if self.lm > self.long_mem_size and prune:
            self.memory_prune()
            self.lm = self.top_k - self.wm * self.num_patches
    
    def memory_read(self, feat, res=True, mem_construct=False):
        '''
        Params:
            - feat: [bs, p, c]
            - mem_k: [bs, t, p, c]
            - mem_v: [bs, t, p, c]
            - mem_c: [bs, t, p, 1]
        '''
        
        affinity = torch.einsum('bpc,bxc->bpx', self.norm_q(feat), self.norm_k(self.mem_k.reshape(self.mem_k.shape[0], -1, self.mem_k.shape[-1])))
        affinity /= torch.sqrt(torch.tensor(feat.shape[-1]).float())
        
        if self.mem_c is not None:
            affinity = affinity * self.mem_c.view(self.mem_c.shape[0], 1, -1)  
        
        attn = torch.softmax(affinity, dim=-1)

        if self.save_attn:
            if self.attn_vis is None:
                self.attn_vis = attn.reshape(-1)
            else:
                self.attn_vis = torch.cat((self.attn_vis, attn.reshape(-1)), dim=0)

        if self.mem_dropout is not None:
            attn = self.mem_dropout(attn)
        
        if self.attn_thresh > 0:
            attn[attn<self.attn_thresh] = 0
            attn = attn / attn.sum(dim=-1, keepdim=True) 
        
        out = torch.einsum('bpx,bxc->bpc', attn, self.norm_v(self.mem_v.reshape(self.mem_v.shape[0], -1, self.mem_v.shape[-1])))
        
        if res:
            out = out + feat
        
        if not mem_construct:
            total_attn = torch.sum(attn, dim=-2)
            self.mem_attn += total_attn[..., None]
        
        return out
    
    def memory_prune(self):

        weights = self.mem_attn / self.mem_count
        weights[self.mem_count<self.work_mem_size+5] = 1e8

        num_mem_b = self.mem_k.shape[1]


        top_k_values, top_k_indices = torch.topk(weights, self.top_k, dim=1)
        top_k_indices_expanded = top_k_indices.expand(-1, -1, self.mem_k.size(-1))


        self.mem_k = torch.gather(self.mem_k, -2, top_k_indices_expanded)
        self.mem_v = torch.gather(self.mem_v, -2, top_k_indices_expanded)
        self.mem_attn = torch.gather(self.mem_attn, -2, top_k_indices)
        self.mem_count = torch.gather(self.mem_count, -2, top_k_indices)
 

        if self.mem_pts is not None:
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, 256, 3)
            self.mem_pts = torch.gather(self.mem_pts, 1, top_k_indices_expanded)
            self.mem_imgs = torch.gather(self.mem_imgs, 1, top_k_indices_expanded)

        num_mem_a = self.mem_k.shape[1]

        print('Memory pruned:', num_mem_b, '->', num_mem_a)
    
    def memory_construct(self, feat_k, feat_v):
        self.mem_k = feat_k.reshape(1, -1, self.mem_k.shape[-1])
        self.mem_v = feat_v.reshape(1, -1, self.mem_v.shape[-1])
        
    

class Spann3R(nn.Module):
    def __init__(self, dus3r_name="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", 
                 use_feat=False, mem_pos_enc=False, memory_dropout=0.15):
        super(Spann3R, self).__init__()
        # config
        self.use_feat = use_feat
        self.mem_pos_enc = mem_pos_enc

        # DUSt3R
        self.dust3r = AsymmetricCroCo3DStereo.from_pretrained(dus3r_name, landscape_only=True)

        # Memory encoder
        self.set_memory_encoder(enc_embed_dim=768 if use_feat else 1024, memory_dropout=memory_dropout) 
        self.set_attn_head()

    
    def normalize(self, img_tensor):
        img_tensor = img_tensor / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_tensor

    def set_memory_encoder(self, enc_depth=6, enc_embed_dim=1024, out_dim=1024, enc_num_heads=16, 
                           mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           memory_dropout=0.15):
        
        self.value_encoder = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, 
                  norm_layer=norm_layer, rope=self.dust3r.rope if self.mem_pos_enc else None)
            for i in range(enc_depth)])
        
        self.value_norm = norm_layer(enc_embed_dim)
        self.value_out = nn.Linear(enc_embed_dim, out_dim)
        
        if not self.use_feat:
            self.pos_patch_embed = copy.deepcopy(self.dust3r.patch_embed)
            self.pos_patch_embed.load_state_dict(self.dust3r.patch_embed.state_dict())
        
        # normalize layers
        self.norm_q = nn.LayerNorm(1024)
        self.norm_k = nn.LayerNorm(1024)
        self.norm_v = nn.LayerNorm(1024)
        self.mem_dropout = nn.Dropout(memory_dropout)
        
    def set_attn_head(self, enc_embed_dim=1024+768, out_dim=1024):
        self.attn_head_1 = nn.Sequential(
            nn.Linear(enc_embed_dim, enc_embed_dim),
            nn.GELU(),
            nn.Linear(enc_embed_dim, out_dim)
        )
        
        self.attn_head_2 = nn.Sequential(
            nn.Linear(enc_embed_dim, enc_embed_dim),
            nn.GELU(),
            nn.Linear(enc_embed_dim, out_dim)
        )

    def encode_image(self, view):
        img = view['img']
        B = img.shape[0]
        im_shape = view.get('true_shape', torch.tensor(img.shape[-2:])[None].repeat(B, 1))
        
        out, pos, _ = self.dust3r._encode_image(img, im_shape)
        
        return out, pos, im_shape
    
    def encode_image_pairs(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']

        B = img1.shape[0]

        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        
        
        out, pos, _ = self.dust3r._encode_image(torch.cat((img1, img2), dim=0),
                                                torch.cat((shape1, shape2), dim=0))
        out, out2 = out.chunk(2, dim=0)
        pos, pos2 = pos.chunk(2, dim=0)
        
        return out, out2, pos, pos2, shape1, shape2
    
    def encode_frames(self, view1, view2, feat1, feat2, pos1, pos2, shape1, shape2):
        if feat1 is None:
            feat1, feat2, pos1, pos2, shape1, shape2 = self.encode_image_pairs(view1, view2)
        
        else:
            feat1, pos1, shape1 = feat2, pos2, shape2
            feat2, pos2, shape2 = self.encode_image(view2)
        
        return feat1, feat2, pos1, pos2, shape1, shape2
    
    def encode_feat_key(self, feat1, feat2, num=1):
        feat = torch.cat((feat1, feat2), dim=-1)
        feat_k = getattr(self, f'attn_head_{num}')(feat)
        
        return feat_k
    
    def encode_value(self, x, pos):
        for block in self.value_encoder:
            x = block(x, pos)
        x = self.value_norm(x)
        x = self.value_out(x)
        return x
    
    def encode_cur_value(self, res1, shape1):
        out, pos_v = self.pos_patch_embed(res1['pts3d'].permute(0, 3, 1, 2), true_shape=shape1)
        cur_v = self.encode_value(out, pos_v)
        
        return cur_v
    
    def decode(self, feat1, pos1, feat2, pos2):
        dec1, dec2 = self.dust3r._decoder(feat1, pos1, feat2, pos2)
        
        return dec1, dec2
    
    def downstream_head(self, dec, true_shape, num=1):
        with torch.amp.autocast('cuda',enabled=False):
            res = self.dust3r._downstream_head(num, [tok.float() for tok in dec], true_shape)

        return res
    
    def find_initial_pair(self, graph, n_frames):
        view1, view2, pred1, pred2 = graph['view1'], graph['view2'], graph['pred1'], graph['pred2']
        n_pairs = len(view1['idx'])

        conf_matrix = torch.zeros(n_frames, n_frames)


        for i in range(n_pairs):
            idx1, idx2 = view1['idx'][i], view2['idx'][i]

            conf1 = pred1['conf'][i]
            conf2 = pred2['conf'][i]

            conf1_sig = (conf1-1)/conf1
            conf2_sig = (conf2-1)/conf2

            conf  = conf1_sig.mean() + conf2_sig.mean()
            conf_matrix[idx1, idx2] = conf

        
        pair_idx = np.unravel_index(conf_matrix.argmax(), conf_matrix.shape)

        print(f'init pair:{pair_idx}, conf: {conf_matrix.max()}')

        return pair_idx
    
    def find_next_best_view(self, frames, idx_todo, feat_fuse, pos1, shape1):
        best_conf = 0.0
        from copy import deepcopy
        for i in idx_todo:
            view = frames[i]
            feat2, pos2, shape2 = self.encode_image(view)
            dec1, dec2 = self.decode(feat_fuse, pos1, feat2, pos2)
            res1 = self.downstream_head(dec1, shape1, 1)
            res2 = self.downstream_head(dec2, shape2, 2)

            conf1 = res1['conf']
            conf2 = res2['conf']

            conf1_sig = (conf1-1)/conf1
            conf2_sig = (conf2-1)/conf2

            
            
            total_conf_mean = conf1_sig.mean() + conf2_sig.mean()


            if total_conf_mean > best_conf:
                best_conf = total_conf_mean
                best_id = i
                best_dec1 = deepcopy(dec1)
                best_dec2 = deepcopy(dec2)
                best_res1 = deepcopy(res1)
                best_res2 = deepcopy(res2)
                best_feat2 = feat2
                best_pos2 = pos2
                best_shape2 = shape2
        

        return best_id, best_dec1, best_dec2, best_res1, best_res2, best_feat2, best_pos2, best_shape2, best_conf

    
    def forward(self, img0, img1, feat0, feat1, pos0, pos1, sp_mem, Q0, mem_construct=False):
        B = img0.shape[0]
        shape0 = torch.tensor(img0.shape[-2:])[None].repeat(B, 1)
        shape1 = torch.tensor(img1.shape[-2:])[None].repeat(B, 1)

        #### Encode frames
        if feat1 is None:
            view1 = {}
            view1['img'] = self.normalize(img1).to('cuda')
            feat1, pos1, shape1 = self.encode_image(view1)

        ##### Memory readout
        if Q0 is not None:
            G0 = sp_mem.memory_read(Q0, res=True, mem_construct=mem_construct)  # eq(2)
        else:
            G0 = feat0

        ##### Decode features
        # dec0[-1]: [bs, p, c=768]
        H0, H1 = self.decode(G0, pos0, feat1, pos1)  # eq(3)

        ##### Regress pointmaps
        with torch.amp.autocast('cuda', enabled=False):
            XC0 = self.downstream_head(H0, shape0, 1)  # eq(5)
            XC1 = self.downstream_head(H1, shape1, 2)
        
        ##### Encode QKV
        Q1 = self.encode_feat_key(feat1, H1[-1], 2)  # eq(4)
        K0 = self.encode_feat_key(feat0, H0[-1], 1)  # eq(6)
        V0 = self.encode_cur_value(XC0, shape0) + K0  # eq(7)
        
        preds = []
        preds.append(XC0)                             
        preds.append(XC1)

        return preds, K0, V0, Q1
    

    def memory_update(self, img0, img1, feat0, feat1, pos0, pos1, sp_mem, Q0, XC0, use_local_mem=False, mem_construct=False):
        B = img0.shape[0]
        shape0 = torch.tensor(img0.shape[-2:])[None].repeat(B, 1)

        #### Encode frames
        if feat1 is None:
            view1 = {}
            view1['img'] = self.normalize(img1).to('cuda')
            feat1, pos1, shape1 = self.encode_image(view1)

        ##### Memory readout
        if use_local_mem:
            G0 = sp_mem.memory_attn(Q0, res=True)
        else:
            if Q0 is not None:
                G0 = sp_mem.memory_read(Q0, res=True, mem_construct=mem_construct)  # eq(2)
            else:
                G0 = feat0

        ##### Decode features
        # dec0[-1]: [bs, p, c=768]
        H0, H1 = self.decode(G0, pos0, feat1, pos1)  # eq(3)
        
        ##### Encode QKV
        Q1 = self.encode_feat_key(feat1, H1[-1], 2)  # eq(4)
        K0 = self.encode_feat_key(feat0, H0[-1], 1)  # eq(6)
        V0 = self.encode_cur_value(XC0, shape0) + K0  # eq(7)
        return K0, V0, Q1
    

    def fill(self, img0, img1):
        
        feat0, feat1, pos0, pos1, shape0, shape1 = None, None, None, None, None, None

        view0 = {}
        view0['img'] = self.normalize(img0).to('cuda')

        view1 = {}
        view1['img'] = self.normalize(img1).to('cuda')

        feat0, feat1, pos0, pos1, shape0, shape1 = self.encode_frames(view0, view1, feat0, feat1, pos0, pos1, shape0, shape1)

        ##### Decode features
        H0, H1 = self.decode(feat0, pos0, feat1, pos1)  # eq(3)

        ##### Regress pointmaps
        with torch.amp.autocast('cuda', enabled=False):
            XC0 = self.downstream_head(H0, shape0, 1)  # eq(5)
            XC1 = self.downstream_head(H1, shape1, 2)
        
        preds = []
        preds.append(XC0)                             
        preds.append(XC1)

        return preds
    
