# generate ground truth
import os
import json
import numpy as np
import roma
import torch
from tqdm import tqdm
from preprocess_utils import TrajStringToMatrix


# given a video, extract all the poses it got
def get_poses(vid, data_path, split):
    frame_path = os.path.join(data_path, split, vid, vid+"_frames")
    traj_file = os.path.join(frame_path, "lowres_wide.traj")
    poses_from_traj = {}
    with open(traj_file) as f:
        trajs = f.readlines()
    for line in trajs:
        traj_timestamp = line.split(" ")[0]
        # align trajection timestamp and frame id and intrinsics
        round_timestamp = f"{round(float(traj_timestamp), 3):.3f}"
        timestamp = round_timestamp
        intrinsic_fn = os.path.join(frame_path, "lowres_wide_intrinsics", f"{vid}_{timestamp}.pincam")
        if not os.path.exists(intrinsic_fn):
            timestamp = f"{float(round_timestamp) - 0.001:.3f}"
            intrinsic_fn = os.path.join(frame_path, "lowres_wide_intrinsics",
                                        f"{vid}_{timestamp}.pincam")
        if not os.path.exists(intrinsic_fn):
            timestamp = f"{float(round_timestamp) + 0.001:.3f}"
            intrinsic_fn = os.path.join(frame_path, "lowres_wide_intrinsics",
                                        f"{vid}_{timestamp}.pincam")
        if not os.path.exists(intrinsic_fn):
            print("traj timestamp", traj_timestamp)
        poses_from_traj[timestamp] = TrajStringToMatrix(line)[1].tolist()

    if os.path.exists(traj_file):
        # self.poses = json.load(open(traj_file))
        poses = poses_from_traj
    else:
        poses = {}
    return poses

def rot_distance(pose1, pose2):
    """Do it in a batch way:
    pose1, pose2: Bx4x4
    """
    rot1 = pose1[:,:3,:3]
    rot2 = pose2[:,:3,:3]
    rot_diff = roma.rotmat_geodesic_distance(rot1, rot2)
    return rot_diff
    
def trans_distance(pose1, pose2):
    """
    Absolute difference
    Do it in a batch way:
    pose1, pose2: Bx4x4
    """
    trans1 = pose1[:, :3, 3]
    trans2 = pose2[:, :3, 3]
    trans_diff = torch.norm(trans1 - trans2, dim=1)
    return trans_diff

def pairwise_diff(poses):
    num_frames = poses.shape[0]
    row_repeat = torch.from_numpy(poses).unsqueeze(0).repeat(num_frames, 1, 1, 1)
    col_repeat = torch.from_numpy(poses).unsqueeze(1).repeat(1, num_frames, 1, 1)
    flatten_r = row_repeat.reshape(-1, 4, 4)
    flatten_c = col_repeat.reshape(-1, 4, 4)
    rot_diff = rot_distance(flatten_r, flatten_c)
    trans_diff = trans_distance(flatten_r, flatten_c)
    rot_diff = rot_diff.reshape(num_frames, num_frames)
    trans_diff = trans_diff.reshape(num_frames, num_frames)
    return rot_diff, trans_diff

def place_adj(rot_diff, trans_diff, rot_thres, trans_thres):
    rot_adj = rot_diff <= rot_thres
    trans_adj = trans_diff <= trans_thres
    place_adj = torch.logical_and(rot_adj, trans_adj)
    return place_adj

# 6 DoF place-place adjacency
def make_data(vid, data_path, data_split, frame_every=10, rot_thres=1.0, trans_thres=1.0):
    """
    take each video, make place-place adjacency graph
    record metadata, p-p adj mat, and 
    """
    poses = get_poses(vid, data_path, data_split)
    frame_ids = list(poses.keys())
    all_poses = [np.array(pose) for pose in poses.values()]
    all_poses = np.stack(all_poses)
    sampled_poses = all_poses[::frame_every,:,:]
    sampled_frame_ids = frame_ids[::frame_every]
    num_sampled_frames = sampled_poses.shape[0]
    rowrepeat = torch.from_numpy(sampled_poses).unsqueeze(0).repeat(num_sampled_frames, 1, 1, 1)
    colrepeat = torch.from_numpy(sampled_poses).unsqueeze(1).repeat(1, num_sampled_frames, 1, 1)
    rot_diff, trans_diff = pairwise_diff(sampled_poses)
    rot_max = rot_diff.max()
    trans_max = trans_diff.max()
    # rot_thres = 1.0 #0.8 # 1.0
    # trans_thres = 1.0 #0.5
    adjcency = place_adj(rot_diff, trans_diff, rot_thres, trans_thres)
    adjcency = adjcency.type(torch.int)
    adjcency = adjcency - torch.eye(adjcency.shape[0], dtype=torch.int)
    # if all connected?
    num_adj = adjcency.sum(dim=1)
    alone_idx = torch.nonzero(num_adj==0).squeeze(dim=0)
    num_alones = alone_idx.size(0)
    ### record ###
    record = {'sampled_frames': sampled_frame_ids}
    meta_record = {
        'rot_max': rot_max.item(), 'trans_max': trans_max.item(), 'num_alones': num_alones, 
        'sample_fps': 10/frame_every, 'rot_thres': rot_thres, 'trans_thres': trans_thres
    }
    adjcency_tolist = adjcency.clone().tolist()
    record['meta'] = meta_record
    record['p-p'] = adjcency_tolist
    ### ### ###
    # p-o
    annotation_file = os.path.join(data_path, data_split, vid, "2d_annotations_"+vid+".json")
    annos = json.load(open(annotation_file, 'r'))
    obj2col = dict()
    uidmap = dict()
    frame2annos = dict()
    
    for frame_id in sampled_frame_ids:
        if frame_id not in annos:
            continue
        anos = annos[frame_id]
        frame2annos[frame_id] = dict()
        for a in anos:
            # skipping small bounding boxes, thresholds: 10 % of H and W
            xmin, ymin, xmax, ymax = a['bbox']
            xx = abs(xmax - xmin)
            yy = abs(ymax - ymin)
            ## -------- ##
            if xx > 26 and yy > 20:
                uid = a["uid"]
                label = a["label"]
                if uid not in obj2col:
                    obj2col[uid] = len(obj2col)
                if label not in uidmap:
                    uidmap[label] = set()
                uidmap[label].add(uid)
                frame2annos[frame_id][uid] = a['bbox']
    for label in uidmap:
        uidmap[label] = list(uidmap[label])
    # build PO adjacency
    num_places = len(sampled_frame_ids)
    num_objs = len(obj2col)
    PO_adc = np.zeros((num_places, num_objs))
    no_obj = []
    all_small = []
    for rid, frame_id in enumerate(sampled_frame_ids):
        # frame has no object bounding boxes
        if frame_id not in annos:
            no_obj.append(frame_id)
            continue # this frame has no object
        # frame has only too small object bounding boxes
        if len(frame2annos[frame_id]) ==0:
            all_small.append(frame_id)
            continue
        annos_frame = annos[frame_id]
        for ano in annos_frame:
            if ano["uid"] in obj2col and ano["uid"] in frame2annos[frame_id]:
                cid = obj2col[ano["uid"]]
                PO_adc[rid, cid] = 1
    # pop out empty entries
    for frame_id in all_small:
        frame2annos.pop(frame_id)
    ### record ###
    record['obj2col'] = obj2col
    record['uidmap'] = uidmap
    record['p-o'] = PO_adc.tolist()
    record['annotations'] = frame2annos
    # meta data recording frame stats
    record['noobj_frames'] = no_obj
    record['allsmall_frames'] = all_small
    return record

def get_gt_po(gt):
        num_frames = len(gt['sampled_frames'])
        num_obj = len(gt['obj2col'])
        frame2row = {frame_id: idx for idx, frame_id in enumerate(gt['sampled_frames'])}
        gt_obj2col = gt['obj2col']
        po_adj = np.zeros((num_frames, num_obj), dtype=int)
        for frame_id in gt['sampled_frames']:
            if frame_id in gt['annotations']:
                anno = gt['annotations'][frame_id]
                for obj_id in anno:
                    po_adj[frame2row[frame_id], gt_obj2col[obj_id]] = 1
        gt['p-o'] = po_adj.tolist()

def regen_po_adj(data_dir, sub_dirs, filtered_video):
    for sub_dir in sub_dirs:
        videos_path = os.path.join(data_dir, sub_dir)
        for videoid in tqdm(os.listdir(videos_path)):
            if videoid in filtered_video:
                continue # skipping invalid videos
            video_path = os.path.join(videos_path, videoid)
            topo_gt_path = os.path.join(video_path, 'refine_topo_gt.json')
            gt = json.load(open(topo_gt_path))
            get_gt_po(gt)
            json.dump(gt, open(topo_gt_path, 'w'))

def get_gt_po(gt):
        num_frames = len(gt['sampled_frames'])
        num_obj = len(gt['obj2col'])
        frame2row = {frame_id: idx for idx, frame_id in enumerate(gt['sampled_frames'])}
        gt_obj2col = gt['obj2col']
        po_adj = np.zeros((num_frames, num_obj), dtype=int)
        for frame_id in gt['sampled_frames']:
            if frame_id in gt['annotations']:
                anno = gt['annotations'][frame_id]
                for obj_id in anno:
                    po_adj[frame2row[frame_id], gt_obj2col[obj_id]] = 1
        gt['p-o'] = po_adj.tolist()

def regen_po_adj(data_dir, sub_dirs, filtered_video):
    for sub_dir in sub_dirs:
        videos_path = os.path.join(data_dir, sub_dir)
        for videoid in tqdm(os.listdir(videos_path)):
            if videoid in filtered_video:
                continue # skipping invalid videos
            video_path = os.path.join(videos_path, videoid)
            topo_gt_path = os.path.join(video_path, 'refine_topo_gt.json')
            gt = json.load(open(topo_gt_path))
            get_gt_po(gt)
            json.dump(gt, open(topo_gt_path, 'w'))
            

if __name__ == "__main__":
    data_path = "/home/arkitdata/"
    data_split = ['Validation', 'Training', 'Test', 'mini-val'] #"Training"
    # vids = os.listdir(os.path.join(data_path, data_split))
    # print(len(vids), "needs to be converted")
    filtered_video = set(["42897846", "42897863", "42897868", "42897871", "47333967", "47204424"])

    regen_po_adj(data_path, data_split, filtered_video)



    # for vid in tqdm(vids):
    #     if vid in filtered_video:
    #         continue
    #     save_path = os.path.join(data_path, data_split, vid, 'refine_topo_gt.json')
    #     # if os.path.exists(save_path):
    #     #     continue # already generated
    #     annotation_file = os.path.join(data_path, data_split, vid, "2d_annotations_"+vid+".json")
    #     if not os.path.exists(annotation_file):
    #         print("SKIPPING because NO 2d annotations available for video {}".format(vid))
    #         continue
    #     gtrec = make_data(vid, data_path, data_split, frame_every=5, rot_thres=1.0, trans_thres=1.0)
        
    #     json.dump(gtrec, open(save_path, 'w'))