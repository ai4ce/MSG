# move data snippets
import os
import shutil
from tqdm import tqdm

# data_dir = "/mnt/NAS/data/zjx/ARKitData/topomap" # "/mnt/NAS/data/zjx/ARKitData/3dod"
# split = "Training"
# split_dir = os.path.join(data_dir, split)
# vids = os.listdir(split_dir)

# correction_data_dir = "corrections_train"

# copy_dir = os.path.join(data_dir, correction_data_dir)
# os.makedirs(copy_dir, exist_ok=True)

# for vid in tqdm(vids):
#     anno = os.path.join(split_dir, vid, f"2d_annotations_{vid}.json")
#     gt = os.path.join(split_dir, vid, "refine_topo_gt.json")
#     shutil.copy(anno, os.path.join(copy_dir, f"{vid}_2d_annotations.json"))
#     shutil.copy(gt, os.path.join(copy_dir, f"{vid}_gt.json"))


# update the data using the corrections
data_dir = "/home/jz4725/topomap/"
split = "Training"
split_dir = os.path.join(data_dir, split)
vids = os.listdir(split_dir)
source_dir = os.path.join(data_dir, "corrections_train")

for vid in tqdm(vids):
    new_anno = os.path.join(source_dir, f"{vid}_2d_annotations.json")
    new_gt = os.path.join(source_dir, f"{vid}_gt.json")
    to_anno = os.path.join(split_dir, vid, f"2d_annotations_{vid}.json")
    to_gt = os.path.join(split_dir, vid, "refine_topo_gt.json")
    shutil.copy(new_anno, to_anno)
    shutil.copy(new_gt, to_gt)
