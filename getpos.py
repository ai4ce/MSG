import os
import cv2
import numpy as np
import json

frame_path = os.path.join("example-data", "41069042")

def get_poses():
    traj_file = os.path.join(frame_path, "41069042_frames", "lowres_wide.traj")
    topo_gt_file = os.path.join(frame_path, "topo_gt.json")
    with open(topo_gt_file) as f:
        topo_gt = json.load(f)
    sampled_frame_ids = topo_gt["sampled_frames"]
    frame_id_set = set(sampled_frame_ids)
    poses_from_traj = {}
    with open(traj_file) as f:
        trajs = f.readlines()
    for line in trajs:
        traj_timestamp = line.split(" ")[0]
        # align trajection timestamp and frame id
        round_timestamp = f"{round(float(traj_timestamp), 3):.3f}"
        timestamp = round_timestamp
        found = False
        if timestamp not in frame_id_set:
            timestamp = f"{float(round_timestamp) - 0.001:.3f}"
        else:
            found = True
        if not found and timestamp not in frame_id_set:
            timestamp = f"{float(round_timestamp) + 0.001:.3f}"
        else:
            found = True
        if not found and timestamp not in frame_id_set:
            # this timestamp is not contained in the processed data
            # print("traj timestamp", traj_timestamp, "")
            found = False
        else:
            found = True
        if found:
            poses_from_traj[timestamp] = TrajStringToMatrix(line)[1][:, 3][:3].tolist()
            # remove if from frame id set
            frame_id_set.remove(timestamp)
    # check if all sampled frames are covered:
    if len(frame_id_set)>0:
        print("Warning: some frames have pose missing!")
        print(frame_id_set)
    return poses_from_traj

# from ARKit Scene, some with modifications
def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)

def scaling(poses):
    """
    scale the poses to display without rounding
    Args:
        poses: a dictionary of poses
    Returns:
        a dictionary of scaled poses
    """
    # find the maximum abs value to determine the scaling factor
    max_abs_val = max(abs(value) for poses in poses.values() for value in poses)

    # find the scaling factor
    scaling_factor = 1000 / max_abs_val

    # scale the poses
    scaled_points = {
        key: [val * scaling_factor for val in values] for key, values in poses.items()
    }
    return scaled_points

def get_xy_coords(poses):
    """
    get the x, y coordinates and round to integer
    Args:
        poses: a dictionary of poses
    Returns:
        a dictionary of x, y coordinates
    """
    xy_coords = {
        key: [round(val[0]), round(val[1])] for key, val in poses.items()
    }
    return xy_coords

print(get_xy_coords(scaling(get_poses())))