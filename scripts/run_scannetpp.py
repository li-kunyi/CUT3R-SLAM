import os
import ast
from glob import glob
from shutil import copyfile
from collections import defaultdict


out = 'outputs/scannetpp'
os.makedirs(f'{out}/meshes', exist_ok=True)
metrics = defaultdict(float)
nrs = ['f34d532901', '39f36da05b', '8b5caf3398', 'b20a261fdf']
seqs = [s for s in sorted(glob("/mnt/home/dataset/scannetpp/*")) if any(n in s for n in nrs)]

for i, seq in enumerate(seqs):
    name = os.path.basename(seq)
    os.makedirs(f'{out}/{name}', exist_ok=True)
    print(name, out)

    # run HI-SLAM2
    cmd = f"python demo.py --imagedir {seq}/color --calib {seq}/calib.txt --cropborder 20 --config config/scannet_config.yaml "
    cmd += f'--output {out}/{name} > {out}/{name}/log.txt'
    if not os.path.exists(f'{out}/{name}/traj_full.txt'):
        os.system(cmd)

    # eval ate
    if not os.path.exists(f'{out}/{name}/ape.txt') or len(open(f'{out}/{name}/ape.txt').readlines()) < 10:
        os.system(f'evo_ape tum {seq}/traj.txt {out}/{name}/traj_kf.txt -vas --no_warnings > {out}/{name}/ape.txt')
    ATE = float([i for i in open(f'{out}/{name}/ape.txt').readlines() if 'rmse' in i][0][-10:-1]) * 100
    metrics['ATE full'] += ATE
    print(f'- full ATE: {ATE:.4f}')

    # eval render
    psnr = ast.literal_eval(open(f'{out}/{name}/psnr/after_opt/final_result_kf.json').read())
    metrics['PSNR'] += psnr['mean_psnr']
    metrics['SSIM'] += psnr['mean_ssim']
    metrics['LPIPS'] += psnr['mean_lpips']
    print(f"- psnr: {psnr['mean_psnr']:.3f}, ssim: {psnr['mean_ssim']:.3f}, lpips: {psnr['mean_lpips']:.3f}")

    # run tsdf fusion
    w = ['5.0','10.0']
    if not os.path.exists(f'{out}/{name}/tsdf_mesh_w{w[-1]}.ply'):
        os.system(f'python tsdf_integrate.py --result {out}/{name} --voxel_size 0.015 --weight {" ".join(w)} > /dev/null')
        for ww in w:
            copyfile(f'{out}/{name}/tsdf_mesh_w{ww}.ply', f'{out}/meshes/{name}_w{ww}.ply')

for r in metrics:
    print(f'{r}: \t {metrics[r]/len(seqs):.4f}')
