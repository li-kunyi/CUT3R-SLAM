# python scripts/run_tum.py

evo_ape tum /home/kunyi/work/data/scannetpp/8b5caf3398/iphone/traj.txt outputs/scannetpp_1102_0/8b5caf3398/traj_kf.txt \
    -vas \
    --save_results outputs/scannetpp_1102_0/8b5caf3398/evo.zip \
    --no_warnings > outputs/scannetpp_1102_0/8b5caf3398/ape.txt

unzip -q outputs/scannetpp_1102_0/8b5caf3398/evo.zip -d outputs/scannetpp_1102_0/8b5caf3398/evo

# python tsdf_integrate.py --result outputs/scannetpp_1102_0/8b5caf3398 --voxel_size 0.015 --weight 5.0 10.0 

# python scripts/eval_recon.py \
#     outputs/scannetpp_1102_0/8b5caf3398/tsdf_mesh_w1.0.ply \
#     /home/kunyi/work/data/scannetpp/8b5caf3398/scans/mesh_aligned_0.05.ply \
#     --eval_3d \
#     --save outputs/scannetpp_1102_0/8b5caf3398/eval_recon_w5.0.txt
