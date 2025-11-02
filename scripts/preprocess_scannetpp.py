import cv2
import os

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
    # Process COLMAP images.txt
    # ----------------------
    colmap_file = os.path.join(base_dir, seq, 'iphone', 'colmap', 'images.txt')
    tum_output_file = os.path.join(os.path.dirname(video_path), "traj.txt")
    
    if not os.path.exists(colmap_file):
        print(f"COLMAP file not found: {colmap_file}")
        continue
    
    poses = []
    with open(colmap_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 9:
                continue  # skip invalid lines

            # Extract pose info
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]

            # Extract frame index from file name, e.g., "frame_000030.jpg" -> 30
            frame_num = int(os.path.splitext(name)[0].split('_')[-1])

            # Format timestamp as 4-digit number (or more if needed)
            timestamp = f"{frame_num:06d}"

            # TUM format: timestamp tx ty tz qx qy qz qw
            poses.append([timestamp, tx, ty, tz, qx, qy, qz, qw])

    # Save to TUM-format file
    with open(tum_output_file, 'w') as f:
        for p in poses:
            f.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
    
    print(f"Saved TUM-format poses to '{tum_output_file}'\n")
