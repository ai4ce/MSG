# report statistics 

import json
import os
import numpy as np
from tqdm import tqdm

data_path = "/mnt/NAS/data/zjx/ARKitData/topomap/"
data_split = "Training"

vids = os.listdir(os.path.join(data_path, data_split))

stats = []

for vid in tqdm(vids):
    video_path = os.path.join(data_path, data_split, vid)
    gtrec = json.load(open(os.path.join(video_path, "refine_topo_gt.json"), 'r'))
    old_gtrec = json.load(open(os.path.join(video_path, "topo_gt.json"), 'r'))
    assert gtrec["sampled_frames"] == old_gtrec["sampled_frames"], (vid)
    num_frames = len(gtrec['sampled_frames'])
    no_object_frames = len(gtrec["noobj_frames"])
    all_small_frames = len(gtrec["allsmall_frames"])
    invalid_frames = no_object_frames + all_small_frames
    percent = np.array([no_object_frames , all_small_frames, invalid_frames])*1.0 / num_frames
    stats.append(percent)

allstats = np.stack(stats, axis=0)
print(allstats.shape, allstats.mean(axis=0))


