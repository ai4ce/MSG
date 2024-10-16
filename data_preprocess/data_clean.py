# dataset clean
import os
import json
import shutil

source_dir = '/home/jz4725/topomap/' # '/mnt/NAS/data/zjx/ARKitData/3dod'
dest_dir =  '/home/jz4725/topomap/' # '/mnt/NAS/data/zjx/ARKitData/topomap'  # parallel directory
sub_dirs = ['Test', 'Validation', 'Training']
target_splits = ['Validation', 'Training', 'Test', 'mini-val']
filtered_video = set(["42897846", "42897863", "42897868", "42897871", "47333967", "47204424"])

from tqdm import tqdm

def copy_required_files(source_path, dest_path, sample_frames, videoid):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    # copy json
    for item in os.listdir(source_path):
        if item.endswith('.json'):
            shutil.copy(os.path.join(source_path, item), dest_path)
    # copy lowres_wide.traj
    frame_path = os.path.join(source_path, videoid+'_frames')
    traj_file_path = os.path.join(frame_path, 'lowres_wide.traj')
    dest_fr_path = os.path.join(dest_path, videoid+'_frames')
    # copy chosen frames
    frames_dir = os.path.join(frame_path, 'lowres_wide')
    dest_frames_dir = os.path.join(dest_fr_path, 'lowres_wide')
    if not os.path.exists(dest_frames_dir):
        os.makedirs(dest_frames_dir)
    for frame_id in sample_frames:
        frame_file = f'{videoid}_{frame_id}.png'
        src_frame_path = os.path.join(frames_dir, frame_file)
        if os.path.exists(src_frame_path):
            shutil.copy(src_frame_path, dest_frames_dir)
        else:
            raise ValueError(videoid, frame_id)
    if os.path.exists(traj_file_path):
        shutil.copy(traj_file_path, dest_fr_path)

def copy_intrinsics_files(source_path, dest_path, videoid):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    frame_path = os.path.join(source_path, videoid+'_frames')
    dest_fr_path = os.path.join(dest_path, videoid+'_frames')

    intrinsics_path = os.path.join(frame_path, 'lowres_wide_intrinsics')
    dest_intrinsics_path = os.path.join(dest_fr_path, 'lowres_wide_intrinsics')

    if os.path.exists(intrinsics_path):
        # if exists, remove and copy
        if os.path.exists(dest_intrinsics_path):
            shutil.rmtree(dest_intrinsics_path)
        shutil.copytree(intrinsics_path, dest_intrinsics_path)

def add_intrinsics(source_dir, dest_dir, sub_dirs, target_splits, filtered_video):
    split_lut = dict()
    for split in target_splits:
        dest_video_path = os.path.join(dest_dir, split)
        split_lut[split] = set(os.listdir(dest_video_path))
    for sub_dir in sub_dirs:
        videos_path = os.path.join(source_dir, sub_dir)
        for videoid in tqdm(os.listdir(videos_path)):
            if videoid in filtered_video:
                continue # skipping invalid videos
            video_path = os.path.join(videos_path, videoid)
            for target_split, lut in split_lut.items():
                if videoid in lut:
                    dest_video_path = os.path.join(dest_dir, target_split, videoid)
                    copy_intrinsics_files(video_path, dest_video_path, videoid)
        


def copy_required_files(source_dir, dest_dir, sub_dirs, filtered_video):
    for sub_dir in sub_dirs:
        videos_path = os.path.join(source_dir, sub_dir)
        for videoid in tqdm(os.listdir(videos_path)):
            if videoid in filtered_video:
                continue # skipping invalid videos
            video_path = os.path.join(videos_path, videoid)
            topo_gt_path = os.path.join(video_path, 'topo_gt.json')
            if os.path.isfile(topo_gt_path):
                with open(topo_gt_path, 'r') as f:
                    dic = json.load(f)
                sample_frames = dic['sampled_frames']
                dest_video_path = os.path.join(dest_dir, sub_dir, videoid)
                copy_required_files(video_path, dest_video_path, sample_frames, videoid)

if __name__ == "__main__":

    # for sub_dir in sub_dirs:
    #     videos_path = os.path.join(source_dir, sub_dir)
    #     for videoid in tqdm(os.listdir(videos_path)):
    #         if videoid in filtered_video:
    #             continue # skipping invalid videos
    #         video_path = os.path.join(videos_path, videoid)
    #         topo_gt_path = os.path.join(video_path, 'topo_gt.json')
    #         if os.path.isfile(topo_gt_path):
    #             with open(topo_gt_path, 'r') as f:
    #                 dic = json.load(f)
    #             sample_frames = dic['sampled_frames']
    #             dest_video_path = os.path.join(dest_dir, sub_dir, videoid)
    #             copy_required_files(video_path, dest_video_path, sample_frames, videoid)
    add_intrinsics(source_dir, dest_dir, sub_dirs, target_splits, filtered_video)


