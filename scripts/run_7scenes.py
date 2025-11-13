import os
import ast
import numpy as np
import open3d as o3d
from collections import defaultdict
from glob import glob
from shutil import copyfile


out = 'outputs/7scenes'
dataset_path="/root/autodl-fs/7scenes/"
os.makedirs(f'{out}/meshes', exist_ok=True)
seqs = [
    "chess",
    "fire",
    "heads",
    "office",
    "pumpkin",
    "redkitchen",
    "stairs",
]

# do not change
kf_every = {
    "chess": -1,
    "fire": 20,
    "heads": 10,
    "office": -1,
    "pumpkin": -1,
    "redkitchen": 10,
    "stairs": 10,
}

metrics = defaultdict(float)
for seq in seqs:
    name = os.path.basename(seq)
    os.makedirs(f'{out}/{name}', exist_ok=True)
    print(name, out)

    # run HI-SLAM2
    # cmd = f'python demo_s.py --imagedir {dataset_path}/{name}/seq-01/color '
    # cmd += f'--config config/7scenes_config.yaml --calib calib/7scenes.txt --output {out}/{name} --kf_every {kf_every[name]}'
    # if not os.path.exists(f'{out}/{name}/traj_full.txt'):
    #     os.system(cmd)

    # # eval ate
    # # os.system(f'evo_ape tum {dataset_path}/{seq}/groundtruth.txt {out}/{name}/traj_full.txt -vas --save_results {out}/{name}/evo.zip --no_warnings > {out}/{name}/ape.txt')
    # os.system(f'evo_ape tum {dataset_path}/{name}/seq-01/{name}.txt {out}/{name}/traj_kf.txt -vas --save_results {out}/{name}/evo.zip --no_warnings > {out}/{name}/ape.txt')
    # os.system(f'unzip -q {out}/{name}/evo.zip -d {out}/{name}/evo')
    # ATE = float([i for i in open(f'{out}/{name}/ape.txt').readlines() if 'rmse' in i][0][-10:-1]) * 100
    # metrics['ATE full'] += ATE
    # print(f'- full ATE: {ATE:.4f}')

    # # eval render
    # psnr = ast.literal_eval(open(f'{out}/{name}/psnr/after_opt/final_result.json').read())
    # print(f"- psnr : {psnr['mean_psnr']:.3f}, ssim: {psnr['mean_ssim']:.3f}, lpips: {psnr['mean_lpips']:.3f}")
    # metrics['PSNR'] += psnr['mean_psnr']
    # metrics['SSIM'] += psnr['mean_ssim']
    # metrics['LPIPS'] += psnr['mean_lpips']

    # eval 3d recon
    cmd = f'python scripts/eval7_scenes_dense.py --dataset /root/autodl-fs/7scenes/{name} '
    cmd += f'--gt /root/autodl-fs/7scenes/{name}/seq-01/{name}.txt '
    cmd += f'--est {out}/{name}/traj_kf.txt '
    cmd += f'--render_path {out}/{name}/renders_kf --no-viz'
    os.system(cmd)

metrics = {'accu': 0, 'comp': 0, 'chamfer': 0}
for name in seqs:
    result = {}
    with open(f'{out}/{name}/renders_kf/3D_eval_results.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split(':', 1)
            result[key.strip()] = float(value.strip())
    metrics['accu'] += result['RMSE_acc']
    metrics['comp'] += result['RMSE_comp']
    metrics['chamfer'] += result['Chamfer_distance']

for r in metrics:
    print(f'{r}: \t {metrics[r]/len(seqs):.4f}')