# converting 3d annotations to 2d annnotaions
# speed up with multi processing
import sys
from pathlib import Path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
import os
import json
import glob
import numpy as np
from typing import List, Tuple, Union
import copy
# import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import time
# from threedod.benchmark_scripts.utils.rotation import convert_angle_axis_to_matrix3
from arkit_utils.tenFpsDataLoader import extract_gt
import arkit_utils.box_utils as box_utils
from shapely.geometry import MultiPoint, box
# from shapely.geometry.linestring import LineString
from shapely.geometry.polygon import Polygon
from preprocess_utils import TrajStringToMatrix, st2_camera_intrinsics, convert_angle_axis_to_matrix3
# from ARKit Scene, some with modifications
# def TrajStringToMatrix(traj_str):
#     """ convert traj_str into translation and rotation matrices
#     Args:
#         traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
#         The file has seven columns:
#         * Column 1: timestamp
#         * Columns 2-4: rotation (axis-angle representation in radians)
#         * Columns 5-7: translation (usually in meters)

#     Returns:
#         ts: translation matrix
#         Rt: rotation matrix
#     """
#     # line=[float(x) for x in traj_str.split()]
#     # ts = line[0];
#     # R = cv2.Rodrigues(np.array(line[1:4]))[0];
#     # t = np.array(line[4:7]);
#     # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
#     tokens = traj_str.split()
#     assert len(tokens) == 7
#     ts = tokens[0]
#     # Rotation in angle axis
#     angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
#     r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
#     # Translation
#     t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
#     extrinsics = np.eye(4, 4)
#     extrinsics[:3, :3] = r_w_to_p
#     extrinsics[:3, -1] = t_w_to_p
#     Rt = np.linalg.inv(extrinsics)
#     return (ts, Rt)


# def st2_camera_intrinsics(filename):
#     w, h, fx, fy, hw, hh = np.loadtxt(filename)
#     return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])

def generate_point(
    # rgb_image,
    depth_image,
    intrinsic,
    subsample=1,
    world_coordinate=True,
    pose=None,
):
    """Generate 3D point coordinates and related rgb feature
    Args:
        rgb_image: (h, w, 3) rgb
        depth_image: (h, w) depth
        intrinsic: (3, 3)
        subsample: int
            resize stride
        world_coordinate: bool
        pose: (4, 4) matrix
            transfer from camera to world coordindate
    Returns:
        points: (N, 3) point cloud coordinates
            in world-coordinates if world_coordinate==True
            else in camera coordinates
        rgb_feat: (N, 3) rgb feature of each point. NOTE: this is removed for efficiency, since not required for our task
    """
    intrinsic_4x4 = np.identity(4)
    intrinsic_4x4[:3, :3] = intrinsic

    u, v = np.meshgrid(
        range(0, depth_image.shape[1], subsample),
        range(0, depth_image.shape[0], subsample),
    )
    d = depth_image[v, u]
    d_filter = d != 0
    mat = np.vstack(
        (
            u[d_filter] * d[d_filter],
            v[d_filter] * d[d_filter],
            d[d_filter],
            np.ones_like(u[d_filter]),
        )
    ) # is mat the same as the points projected 2d? un normalized
    pcd_2d = np.vstack(
        (
            u[d_filter],
            v[d_filter],
        )
    )
    new_points_3d = np.dot(np.linalg.inv(intrinsic_4x4), mat)[:3]
    if world_coordinate:
        new_points_3d_padding = np.vstack(
            (new_points_3d, np.ones((1, new_points_3d.shape[1])))
        )
        world_coord_padding = np.dot(pose, new_points_3d_padding)
        new_points_3d = world_coord_padding[:3]

    # rgb_feat = rgb_image[v, u][d_filter]

    return new_points_3d.T, pcd_2d


# from mmdet3d
def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        if not isinstance(img_intersection, Polygon):
            return None
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def get_2d_box(box_corners_3d, frame_pose, intrinsics, image_shape):
    box_corner_pad = np.vstack(
        (box_corners_3d, np.ones((1, box_corners_3d.shape[1])))
    )
    corners_proj = np.dot(np.linalg.inv(frame_pose), box_corner_pad)[:3]
    # Filter out the corners that are not in front of the calibrated
    # sensor. From mmdet3d
    in_front = np.argwhere(corners_proj[2, :] > 0).flatten()
    corners_proj = corners_proj[:, in_front]
    nbr_points = corners_proj.shape[1]
    # print(nbr_points)
    corners_proj_pad = np.vstack(
        (corners_proj, np.ones((1, corners_proj.shape[1])))
    )
    intrinsic_pad = np.identity(4)
    intrinsic_pad[:3, :3] = intrinsics
    corners_2d_pad = np.dot(intrinsic_pad, corners_proj_pad)
    corners_2d_pad = corners_2d_pad[:3, :]
    # print(corners_2d_pad.shape)
    corners_2d_normalized = corners_2d_pad / corners_2d_pad[2:3, :].repeat(3,0).reshape(3, nbr_points)
    corners_2d = corners_2d_normalized.T[:,:2].tolist()
    # post
    # print("2d", len(corners_2d))
    corners_2d_final = post_process_coords(corners_2d, image_shape[:2])
    return corners_2d_final


def convert_annotation_2d_bbox(video_id, data_path, data_split):
    '''Convert bbox annotations from 3d to 2d. ONLY operates on bbox corners'''
    root_path = os.path.join(data_path, data_split, video_id, video_id+"_frames")
    traj_path = os.path.join(root_path, "lowres_wide.traj")
    # print(video_id, traj_path)
    anno_path = os.path.join(data_path, data_split, video_id, video_id+"_3dod_annotation.json")
    skipped, boxes_corners, centers, sizes, labels, uids = extract_gt(anno_path)
    image_folder = os.path.join(root_path, "lowres_wide")
    image_shape = (192, 256, 3)
    if not os.path.exists(image_folder):
        frame_ids = []
    else:
        rgb_images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
        frame_ids = [os.path.basename(x) for x in rgb_images]
    frame_ids = [x.split(".png")[0].split("_")[1] for x in frame_ids]
    frame_ids.sort()
    
    with open(traj_path) as f:
        traj = f.readlines()
    # convert traj to json dict
    poses_from_traj = {}
    for line in traj:
        traj_timestamp = line.split(" ")[0]
        # sync pose timestamp and intrinsics
        round_timestamp = f"{round(float(traj_timestamp), 3):.3f}"
        timestamp = round_timestamp
        intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics", f"{video_id}_{timestamp}.pincam")
        if not os.path.exists(intrinsic_fn):
            timestamp = f"{float(round_timestamp) - 0.001:.3f}"
            intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics",
                                        f"{video_id}_{timestamp}.pincam")
        if not os.path.exists(intrinsic_fn):
            timestamp = f"{float(round_timestamp) + 0.001:.3f}"
            intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics",
                                        f"{video_id}_{timestamp}.pincam")
        if not os.path.exists(intrinsic_fn):
            print("traj timestamp", traj_timestamp)
        poses_from_traj[timestamp] = TrajStringToMatrix(line)[1].tolist()
    poses = poses_from_traj
    
    intrinsics = {}
    for frame_id in frame_ids:
        intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics", f"{video_id}_{frame_id}.pincam")
        if not os.path.exists(intrinsic_fn):
            intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics",
                                        f"{video_id}_{float(frame_id) - 0.001:.3f}.pincam")
        if not os.path.exists(intrinsic_fn):
            intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics",
                                        f"{video_id}_{float(frame_id) + 0.001:.3f}.pincam")
        if not os.path.exists(intrinsic_fn):
            print("frame_id", frame_id, "intrinsic fn", intrinsic_fn)
        intrinsics[frame_id] = st2_camera_intrinsics(intrinsic_fn)
        
    filtered_frame_ids = []
    video_annotations = {}
    for frame_id in frame_ids:
            
        frame_pose = np.array(poses[frame_id]) # extrinsics
        frame_intrinsics = intrinsics[frame_id] # intrinsics
        has_object = False
        frame_annotations = []
        for obj in range(len(labels)):
            label = labels[obj]
            uid = uids[obj]
            box_corners_3d = boxes_corners[obj].T
            bbox_2d = get_2d_box(box_corners_3d, frame_pose, frame_intrinsics, image_shape)
            if bbox_2d is None:
                # this obj does not appear in this frame
                continue
            else:
                annotation_2d = {'label': label, 'uid': uid, 'bbox': bbox_2d} #(min_x, min_y, max_x, max_y)
                frame_annotations.append(annotation_2d)
                has_object = True
        if has_object:
            filtered_frame_ids.append(frame_id)
            video_annotations[frame_id] = frame_annotations
    return filtered_frame_ids, video_annotations, len(frame_ids)


def convert_annotation_2d_points(video_id, data_path, data_split):
    '''Convert bbox annotations from 3d to 2d. ONLY operates on 2d points'''
    root_path = os.path.join(data_path, data_split, video_id, video_id+"_frames")
    anno_path = os.path.join(data_path, data_split, video_id, video_id+"_3dod_annotation.json")
    skipped, boxes_corners, centers, sizes, labels, uids = extract_gt(anno_path)
    assert len(labels) == len(uids)
    uid_map = dict(zip(uids, labels))
    if skipped or boxes_corners.shape[0] == 0:
        return (None, None, None)
    
    loader = Convert2DDataLoader(
        root_path=root_path,
    )
    video_annotations = {}
    filtered_frame_ids = []
    # frame_rate = max(2, len(loader) // 300)
    for idx in range(len(loader)):
        frame = loader[idx]
        image_path = frame["image_path"]
        # frame_id = image_path.split(".")[-2]
        # frame_id = frame_id.split("_")[-1]
        frame_id = frame["frame_id"]
        pcd = frame["pcd"]
        pcd_2d = frame["pcd_2d"] #(n,2)
        offset_h = frame["offset_H"] # 0 or 64 for 256
        offset_w = frame["offset_W"] # 0 or 48 for 192
        # has_object = False
        # filter
        # keep the points inside the actual image
        filter_H = np.logical_and(pcd_2d[:,0]>=0+offset_h, pcd_2d[:,0]<256+offset_h)
        filter_W = np.logical_and(pcd_2d[:,1]>=0+offset_w, pcd_2d[:,1]<192+offset_w)
        filter_image = filter_H & filter_W
        # (n,3) -> (n',3), (n,2) -> (n', ), n' for in the image
        pcd_in_image = pcd[filter_image]
        pcd_2d_in_image = pcd_2d[filter_image]

        # 2.3 apply a simple box-filter by removing boxes with < 20 points
        # (n', m)
        mask_pts_in_box = box_utils.points_in_boxes(pcd_in_image, boxes_corners)
        # (m, )
        pts_cnt = np.sum(mask_pts_in_box, axis=0)
        mask_box = pts_cnt > 20 # keep the boxes that contain more than 20 points
        if np.sum(mask_box) == 0:
            #has_object = False
            continue # this frame has no object
        # if has object
        mask_pts_in_valid_box = mask_pts_in_box[:, mask_box] #(n', m')
        # box_cnt = np.sum(mask_pts_in_valid_box, axis=1)
        # mask_point = box_cnt>0 # keep the points that are in at least one of those valid boxes
        # points_keep_2d = pcd_2d_in_image[mask_point]
        labels_keep = np.array(labels)[mask_box]
        labels_keep = labels_keep.tolist()
        uids_keep = np.array(uids)[mask_box]
        frame_annotations = []
        filtered_frame_ids.append(frame_id)
        for oid in range(mask_pts_in_valid_box.shape[1]):
            label = labels_keep[oid]
            uid = uids_keep[oid]
            assert uid_map[uid] == label, (uid, label, mask_box)
            box_pts = pcd_2d_in_image[mask_pts_in_valid_box[:, oid]] - np.array([offset_h, offset_w])[None, :] # offset to image size
            xmin = int(min(box_pts[:,0]))
            ymin = int(min(box_pts[:,1]))
            xmax = int(max(box_pts[:,0]))
            ymax = int(max(box_pts[:,1]))
            bbox_2d = [xmin, ymin, xmax, ymax]
            # print(bbox_2d, type(bbox_2d[0]))
            annotation_2d = {'label': label, 'uid': uid, 'bbox': bbox_2d} #(min_x, min_y, max_x, max_y)
            frame_annotations.append(annotation_2d)
        video_annotations[frame_id] = frame_annotations
    return filtered_frame_ids, video_annotations, len(loader)

# modify from the original ARKitScene's TenFpsDataLoader
class Convert2DDataLoader(object):
    def __init__(
        self,
        root_path,
        frame_rate=1,
        with_color_image=True,
        subsample=2,
        world_coordinate=True,
    ):
        """
        Args:
            root_path: path with all info for a scene_id
                color, color_2det, depth, label, vote, ...
            an2d_root: path to scene_id.json
                or None
            logger:
            frame_rate: int
            subsample: int
            world_coordinate: bool
        """
        self.root_path = root_path

        # pipeline does box residual coding here
        # self.num_class = len(class_names)

        # self.dc = ARKitDatasetConfig()

        depth_folder = os.path.join(self.root_path, "lowres_depth")
        if not os.path.exists(depth_folder):
            # print("WARNING: depth folder not exist")
            self.frame_ids = []
        else:
            # print("foound depth folder", depth_folder)
            depth_images = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
            self.frame_ids = [os.path.basename(x) for x in depth_images]
            self.frame_ids = [x.split(".png")[0].split("_")[1] for x in self.frame_ids]
            self.video_id = depth_folder.split('/')[-3]
            self.frame_ids = [x for x in self.frame_ids]
            self.frame_ids.sort()
            self.intrinsics = {}

        traj_file = os.path.join(self.root_path, 'lowres_wide.traj')
        with open(traj_file) as f:
            self.traj = f.readlines()
        # convert traj to json dict
        poses_from_traj = {}
        for line in self.traj:
            traj_timestamp = line.split(" ")[0]
            # align trajection timestamp and frame id
            round_timestamp = f"{round(float(traj_timestamp), 3):.3f}"
            timestamp = round_timestamp
            intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics", f"{self.video_id}_{timestamp}.pincam")
            if not os.path.exists(intrinsic_fn):
                timestamp = f"{float(round_timestamp) - 0.001:.3f}"
                intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics",
                                            f"{self.video_id}_{timestamp}.pincam")
            if not os.path.exists(intrinsic_fn):
                timestamp = f"{float(round_timestamp) + 0.001:.3f}"
                intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics",
                                            f"{self.video_id}_{timestamp}.pincam")
            if not os.path.exists(intrinsic_fn):
                print("traj timestamp", traj_timestamp)
            poses_from_traj[timestamp] = TrajStringToMatrix(line)[1].tolist()

        if os.path.exists(traj_file):
            # self.poses = json.load(open(traj_file))
            self.poses = poses_from_traj
        else:
            self.poses = {}

        # get intrinsics
        for frame_id in self.frame_ids:
            intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics", f"{self.video_id}_{frame_id}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics",
                                            f"{self.video_id}_{float(frame_id) - 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics",
                                            f"{self.video_id}_{float(frame_id) + 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                print("frame_id", frame_id)
                print(intrinsic_fn)
            self.intrinsics[frame_id] = st2_camera_intrinsics(intrinsic_fn)


        self.frame_rate = frame_rate
        self.subsample = subsample
        self.with_color_image = with_color_image
        self.world_coordinate = world_coordinate


    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        """
        Returns:
            frame: a dict
                {frame_id}: str
                {depth}: (h, w)
                {image}: (h, w)
                {image_path}: str
                {intrinsics}: np.array 3x3
                {pose}: np.array 4x4
                {pcd}: np.array (n, 3)
                    in world coordinate
                {pcd_2d}: (n, 2), in image
        """
        frame_id = self.frame_ids[idx]
        frame = {}
        frame["frame_id"] = frame_id
        fname = "{}_{}.png".format(self.video_id, frame_id)
        # fname = "{}.png".format(frame_id)
        depth_image_path = os.path.join(self.root_path, "lowres_depth", fname)
        if not os.path.exists(depth_image_path):
            print(depth_image_path)

        image_path = os.path.join(self.root_path, "lowres_wide", fname)

        if not os.path.exists(depth_image_path):
            print(depth_image_path, "does not exist")
        frame["depth"] = cv2.imread(depth_image_path, -1)
        frame["image"] = cv2.imread(image_path)
        frame["image_path"] = image_path
        depth_height, depth_width = frame["depth"].shape
        im_height, im_width, im_channels = frame["image"].shape

        frame["intrinsics"] = copy.deepcopy(self.intrinsics[frame_id])
        if str(frame_id) in self.poses.keys():
            frame_pose = np.array(self.poses[str(frame_id)])
        else:
            for my_key in list(self.poses.keys()):
                if abs(float(frame_id) - float(my_key)) < 0.005:
                    frame_pose = np.array(self.poses[str(my_key)])
        frame["pose"] = copy.deepcopy(frame_pose)

        # im_height_scale = np.float(depth_height) / im_height
        # im_width_scale = np.float(depth_width) / im_width

        offset_H = 0
        offset_W = 0
        if depth_height != im_height:
            frame["image"] = np.zeros([depth_height, depth_width, 3])  # 288, 384, 3
            frame["image"][48 : 48 + 192, 64 : 64 + 256, :] = cv2.imread(image_path)
            offset_H = 64
            offset_W = 48

        # (m, n, _) = frame["image"].shape
        depth_image = frame["depth"] / 1000.0
        # rgb_image = frame["image"] / 255.0

        pcd, pcd_2d = generate_point(
            # rgb_image,
            depth_image,
            frame["intrinsics"],
            self.subsample,
            self.world_coordinate,
            frame_pose,
        )
        # build pcd filter, only keep those in the acutal image for our 2d task
        # also need to take care of coordinates later
        frame["offset_H"] = offset_H
        frame["offset_W"] = offset_W
        pcd_2d = pcd_2d.T

        frame["pcd"] = pcd
        frame["pcd_2d"] = pcd_2d
        return frame


def multi_work(video_ids):
    print("handling", len(video_ids), "videos from video", video_ids[0], video_ids[-1])
    data_path = "/home/arkitdata/"
    data_split = "Training"
    subset_stats = {
        "total_frames":0,
        "total_filtered":0
    }
    for idx, video_id in enumerate(video_ids):
        anno_path = os.path.join(data_path, data_split, video_id, "2d_annotations_"+video_id+".json")
        # if os.path.exists(anno_path):
            # print("skipping", video_id)
            # continue # skipping generated data
        filtered_frame_ids, video_annotations, num_frames = convert_annotation_2d_points(video_id, data_path, data_split)
        if filtered_frame_ids is None:
            print("NO GOOD FRAMES for VIDEO", video_id, "skip and continue")
            continue
        # subset_stats.append(len(filtered_frame_ids)/num_frames*100)
        subset_stats["total_filtered"] += len(filtered_frame_ids)
        subset_stats["total_frames"] += num_frames
        # dump 2d annotations
        json.dump(video_annotations, open(anno_path, 'w'))
        if idx % 10 ==0:
            print("progress made by subprocess:", idx/len(video_ids), "curr", video_id, ">>>", video_ids[-1])
    print("work finished from ", video_ids[0], "to", video_ids[-1])
    return subset_stats



if __name__ == "__main__":
    data_path = "/home/arkitdata/"
    data_split = "Training"
    videos = os.listdir(os.path.join(data_path, data_split))
    print(len(videos), "needs to be converted")
    videos_left = []
    for vid in videos:
        anno_path = os.path.join(data_path, data_split, vid, "fixed_2d_annotations_"+vid+".json")
        if not os.path.exists(anno_path):
            videos_left.append(vid)
    videos = videos_left
    print(len(videos), "left to be converted")
    # stats = []
    # total_filtered = 0
    # total_frames = 0
    t = time.time()
    # for video_id in videos:
    #     print("video", video_id)
    #     filtered_frame_ids, video_annotations, num_frames = convert_annotation_2d_points(video_id, data_path, data_split)
    #     if filtered_frame_ids is None:
    #         print("NO GOOD FRAMES for VIDEO", video_id, "skip and continue")
    #         continue
    #     stats.append(len(filtered_frame_ids)/num_frames*100)
    #     total_filtered += len(filtered_frame_ids)
    #     total_frames += num_frames
    #     # dump 2d annotations
    #     anno_path = os.path.join(data_path, data_split, video_id, "2d_annotations_"+video_id+".json")
    #     # anno_fname = video_id+".json"
    #     # anno_path = os.path.join(annotation_dir, anno_fname)
    #     json.dump(video_annotations, open(anno_path, 'w'))
    num_procs = 6
    num_videos = len(videos)
    work = num_videos // num_procs + 1
    video_partitions = []
    for i in range(num_procs):
        video_partitions.append(videos[i*work:min(num_videos, (i+1)*work)])
    with Pool(num_procs) as p:
        print(p.map(multi_work, video_partitions))
    # print("statistics for", data_split)
    elapased = time.time() - t
    print("total time: %f sec" % elapased)
    # print(total_filtered, "/", total_frames, f"{total_filtered/total_frames*100:.2f}%")
    # print("for each video on average:", np.asarray(stats).mean(), "%")


        