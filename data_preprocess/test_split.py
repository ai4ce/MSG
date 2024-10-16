import os
import shutil
# split test set from the original Validation
source_dir = '/mnt/NAS/data/zjx/ARKitData/topomap/Validation'
dest_dir = '/mnt/NAS/data/zjx/ARKitData/topomap/Test'


if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)


video_ids = sorted(os.listdir(source_dir))


selected_video_ids = video_ids[:200]

for videoid in selected_video_ids:
    source_path = os.path.join(source_dir, videoid)
    dest_path = os.path.join(dest_dir, videoid)
    shutil.move(source_path, dest_path)

print(f"move first 200 videos from '{source_dir}' to '{dest_dir}'.")
