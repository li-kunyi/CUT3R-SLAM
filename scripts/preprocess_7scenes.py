import os
import shutil
from glob import glob

def organize_7scenes(root_dir):
    scenes = sorted(os.listdir(root_dir))
    for scene in scenes:
        scene_path = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        
        seqs = sorted(os.listdir(scene_path))
        for seq in seqs:
            seq_path = os.path.join(scene_path, seq)
            if not os.path.isdir(seq_path):
                continue
            
            print(f"Processing {seq_path} ...")
            
            out_color = os.path.join(seq_path, "color")
            out_depth = os.path.join(seq_path, "depth")
            out_pose = os.path.join(seq_path, "pose")
            os.makedirs(out_color, exist_ok=True)
            os.makedirs(out_depth, exist_ok=True)
            os.makedirs(out_pose, exist_ok=True)

            for f in glob(os.path.join(seq_path, "*color.png")):
                shutil.copy2(f, os.path.join(out_color, os.path.basename(f)))
            
            for f in glob(os.path.join(seq_path, "*depth.png")):
                shutil.copy2(f, os.path.join(out_depth, os.path.basename(f)))
            
            for f in glob(os.path.join(seq_path, "*pose.txt")):
                shutil.copy2(f, os.path.join(out_pose, os.path.basename(f)))


if __name__ == "__main__":
    root_dir = "/home/kunyi/work/data/7-scenes"
    organize_7scenes(root_dir)
