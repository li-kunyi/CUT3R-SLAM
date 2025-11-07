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
    # "chess",
    # "fire",
    # "heads",
    # "office",
    # "pumpkin",
    "redkitchen",
    "stairs",
]

metrics = defaultdict(float)
for seq in seqs:
    name = os.path.basename(seq)
    os.makedirs(f'{out}/{name}', exist_ok=True)
    print(name, out)

    # run HI-SLAM2
    cmd = f'python demo_s.py --imagedir {dataset_path}/{name}/seq-01/color '
    cmd += f'--config config/7scenes_config.yaml --calib calib/7scenes.txt --output {out}/{name}'
    if not os.path.exists(f'{out}/{name}/traj_full.txt'):
        os.system(cmd)

    # eval ate
    # os.system(f'evo_ape tum {dataset_path}/{seq}/groundtruth.txt {out}/{name}/traj_full.txt -vas --save_results {out}/{name}/evo.zip --no_warnings > {out}/{name}/ape.txt')
    os.system(f'evo_ape tum {dataset_path}/{name}/seq-01/{name}.txt {out}/{name}/traj_kf.txt -vas --save_results {out}/{name}/evo.zip --no_warnings > {out}/{name}/ape.txt')
    os.system(f'unzip -q {out}/{name}/evo.zip -d {out}/{name}/evo')
    ATE = float([i for i in open(f'{out}/{name}/ape.txt').readlines() if 'rmse' in i][0][-10:-1]) * 100
    metrics['ATE full'] += ATE
    print(f'- full ATE: {ATE:.4f}')

    # # eval render
    # psnr = ast.literal_eval(open(f'{out}/{name}/psnr/after_opt/final_result.json').read())
    # print(f"- psnr : {psnr['mean_psnr']:.3f}, ssim: {psnr['mean_ssim']:.3f}, lpips: {psnr['mean_lpips']:.3f}")
    # metrics['PSNR'] += psnr['mean_psnr']
    # metrics['SSIM'] += psnr['mean_ssim']
    # metrics['LPIPS'] += psnr['mean_lpips']

    # # run tsdf fusion
    # w = 2
    # weight = f'w{w:.1f}'
    # if not os.path.exists(f'{out}/{name}/tsdf_mesh_{weight}.ply'):
    #     os.system(f'python tsdf_integrate.py --result {out}/{name} --voxel_size 0.006 --weight {w} > /dev/null')
    #     ply = o3d.io.read_triangle_mesh(f'{out}/{name}/tsdf_mesh_{weight}.ply')
    #     ply = ply.transform(np.load(f'{out}/{name}/evo/alignment_transformation_sim3.npy'))
    #     o3d.io.write_triangle_mesh(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply', ply)
    #     copyfile(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply', f'{out}/meshes/{name}.ply')
    
    # # eval 3d recon
    # if not os.path.exists(f'{out}/{name}/eval_recon_{weight}.txt'):
    #     os.system(f'python scripts/eval_recon.py {out}/{name}/tsdf_mesh_{weight}_aligned.ply data/Replica/gt_mesh_culled/{name}.ply --eval_3d --save {out}/{name}/eval_recon_{weight}.txt > /dev/null')
    # result = ast.literal_eval(open(f'{out}/{name}/eval_recon_{weight}.txt').read())
    # metrics['accu'] += result['mean precision']
    # metrics['comp'] += result['mean recall']
    # metrics['compr'] += result['recall']
    # print(f"- acc: {result['mean precision']:.3f}, comp: {result['mean recall']:.3f}, comp rat: {result['recall']:.3f}\n")

# for r in metrics:
#     print(f'{r}: \t {metrics[r]/len(seqs):.4f}')
