import os
import ast
from glob import glob
from shutil import copyfile
from collections import defaultdict
# import numpy as np
# import open3d as o3d

base_dir = '/root/autodl-fs/scannetpp'
out = 'outputs/scannetpp'
os.makedirs(f'{out}/meshes', exist_ok=True)
metrics = defaultdict(float)
nrs = [
    '8b5caf3398', 
    '39f36da05b', 
    'f34d532901', 
    '0e75f3c4d9',
    ]
# nrs = ['39f36da05b']

seqs = [s for s in sorted(glob(f"{base_dir}/*")) if any(n in s for n in nrs)]

for i, seq in enumerate(seqs):
    name = os.path.basename(seq)
    os.makedirs(f'{out}/{name}', exist_ok=True)
    print(name, out)

    # run HI-SLAM2
    cmd = f"python demo_s.py --imagedir {seq}/iphone/color --calib {seq}/iphone/calib.txt --config config/scannetpp_config.yaml "
    cmd += f'--output {out}/{name} > {out}/{name}/log.txt'
    if not os.path.exists(f'{out}/{name}/traj_full.txt'):
        os.system(cmd)

    # eval ate
    os.system(f'evo_ape tum {seq}/iphone/traj.txt {out}/{name}/traj_kf.txt -vas --save_results {out}/{name}/evo.zip --no_warnings > {out}/{name}/ape.txt')
    os.system(f'unzip -q {out}/{name}/evo.zip -d {out}/{name}/evo')
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
    # w = 1
    # weight = f'w{w:.1f}'
    # if not os.path.exists(f'{out}/{name}/tsdf_mesh_{weight}.ply'):
    #     os.system(f'python tsdf_integrate.py --result {out}/{name} --voxel_size 0.01 --weight {w} > /dev/null')
    #     ply = o3d.io.read_triangle_mesh(f'{out}/{name}/tsdf_mesh_{weight}.ply')
    #     ply = ply.transform(np.load(f'{out}/{name}/evo/alignment_transformation_sim3.npy'))
    #     o3d.io.write_triangle_mesh(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply', ply)
    #     copyfile(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply', f'{out}/meshes/{name}.ply')
    
    # # eval 3d recon
    # if not os.path.exists(f'{out}/{name}/eval_recon_{weight}.txt'):
    #     os.system(f'python scripts/eval_recon.py {out}/{name}/tsdf_mesh_{weight}_aligned.ply {base_dir}/{name}/scans/mesh_aligned_0.05.ply --eval_3d --save {out}/{name}/eval_recon_{weight}.txt > /dev/null')
    # result = ast.literal_eval(open(f'{out}/{name}/eval_recon_{weight}.txt').read())
    # metrics['accu'] += result['mean precision']
    # metrics['comp'] += result['mean recall']
    # metrics['compr'] += result['recall']
    # print(f"- acc: {result['mean precision']:.3f}, comp: {result['mean recall']:.3f}, comp rat: {result['recall']:.3f}\n")


for r in metrics:
    print(f'{r}: \t {metrics[r]/len(seqs):.4f}')
