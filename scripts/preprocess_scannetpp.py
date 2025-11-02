import cv2
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Base data folder and sequences
base_dir = '/home/kunyi/work/data/scannetpp'
seqs = ['f34d532901', '39f36da05b', '8b5caf3398', 'b20a261fdf']

# Process each sequence
for seq in seqs:
    print(f"Processing sequence: {seq}")
    
    video_path = os.path.join(base_dir, seq, 'iphone', 'rgb.mp4')
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        continue
    
    # Create color folder
    output_folder = os.path.join(os.path.dirname(video_path), "color")
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        continue
    
    frame_count = 0
    saved_count = 0
    
    # Save every 10th frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 10 == 0:
            output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Saved {saved_count} frames to '{output_folder}'")
    
    # ----------------------
    # Process pose_intrinsic_imu.json
    # ----------------------
    pose_file = os.path.join(base_dir, seq, 'iphone', 'pose_intrinsic_imu.json')
    tum_output_file = os.path.join(os.path.dirname(video_path), "traj.txt")
    intrinsic_output_file = os.path.join(os.path.dirname(video_path), "calib.txt")
    
    if not os.path.exists(pose_file):
        print(f"Pose file not found: {pose_file}")
        continue
    
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    
    poses = []
    frame_idx = 0
    for i, key in enumerate(sorted(pose_data.keys())):
        if i % 10 != 0:
            continue  # skip frames, take every 10th
        
        pose_entry = pose_data[key]
        aligned_pose = np.array(pose_entry['aligned_pose'])  # 4x4
        
        # Extract rotation R and translation t
        R = aligned_pose[:3,:3]
        t = aligned_pose[:3,3]
        
        # Convert rotation matrix to quaternion (x, y, z, w)
        r = R_scipy.from_matrix(R)
        qx, qy, qz, qw = r.as_quat()  # [x, y, z, w]
        
        # TUM format: timestamp tx ty tz qx qy qz qw
        timestamp = f"{frame_idx:06d}"
        poses.append([timestamp, t[0], t[1], t[2], qx, qy, qz, qw])
        
        frame_idx += 10  # increment by 10 for skipped frames
    
    # Save TUM-format poses
    with open(tum_output_file, 'w') as f:
        for p in poses:
            f.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
    
    print(f"Saved TUM-format poses to '{tum_output_file}'")
    
    # ----------------------
    # Extract intrinsic
    # ----------------------
    # Take intrinsic from the first frame
    first_key = sorted(pose_data.keys())[0]
    intrinsic = pose_data[first_key]['intrinsic']
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]
    
    # Save intrinsic to txt
    with open(intrinsic_output_file, 'w') as f:
        f.write(f"{fx} {fy} {cx} {cy}\n")
    
    print(f"Saved intrinsic parameters to '{intrinsic_output_file}'\n")
