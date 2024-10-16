import sys
import json
import os
import cv2
import numpy as np
import shutil
import re


DISTANCE_THRESHOLD = 10
SCALE_RATIO = 1000
RADIUS_EXTENSION = 40



def adjacency_matrix_to_list(adj_matrix) -> dict:
    adj_list = {}
    for i, row in enumerate(adj_matrix):
        adj_list[i] = []
        for j, val in enumerate(row):
            if val != 0:
                adj_list[i].append(j)
    return adj_list

def adjacency_list_to_edges(adj_list) -> list:
    edges = []
    for source, targets in adj_list.items():
        for target in targets:
            edges.append((source, target))
    return edges

def get_camera_coords_scaled_adjusted(graph, traj_file) -> dict:
    sampled_frames_id = graph["sampled_frames"]
    poses = get_poses(sampled_frames_id, traj_file)
    poses_scaled = scaling(poses)
    poses_scaled_adjusted = coord_adjustment(poses_scaled)
    return poses_scaled_adjusted

# image nodes
"""
"nodes": [
    {
        "id": "cam0",
        "nodeType": "camera",           // node type
        "sampled_frames": "3044.239",   // sampled frame id
        "label": "0",                 // node label
        "x": 0,                       // node x coordinate
        "y": 0,                       // node y coordinate
        "annotations": {            // annotations
            "NB59gmIiC4u5h2Mw": [   // object annotation
                48,
                110,
                236,
                190
            ],
            "RnVg7UM3yU93OL1o": [
                100,
                24,
                184,
                112
            ]
        }
    },
],
"""
def get_camera_nodes(graph, pp_adj_list: dict, poses_scaled_adjusted: dict):
    nodes = []
    sampled_frames_id = graph["sampled_frames"]

    for node in pp_adj_list.keys():
        sampled_frames = sampled_frames_id[node]
        # If the sampled frame of the node is in the annotation, add the annotation to the node information
        if sampled_frames in graph["annotations"]:
            annotations = graph["annotations"][sampled_frames]
        else:
            annotations = None
        nodes.append({"id": "cam" + str(node),
                        "nodeType": "camera",
                        "sampled_frames": sampled_frames,
                        "label": str(node),
                        "x": poses_scaled_adjusted[sampled_frames][0], "y": poses_scaled_adjusted[sampled_frames][1],
                        "annotations": annotations})
    return nodes

# object nodes, needs x,y for display
"""
"nodes": [
    {
        "id": "obj0",
        "nodeType": "object",          // node type
        "label": "0",                  // node label
        "objUid": "NB59gmIiC4u5h2Mw",   // object uid
        "objName": "bed",              // object name
        "x": 0,                        // node x, y coordinate
        "y": 0                         // obtained by computing
    },
],
"""
def get_object_nodes(graph):
    nodes = []
    uidmap = graph["uidmap"]    # map from object name to object uids
    for objName, objUids in uidmap.items():
        for objUid in objUids:
            objCol = graph["obj2col"][objUid]   # object column, corresponding to the object uid
            nodes.append({"id": "obj" + str(objCol),
                          "nodeType": "object",
                          "label": str(objCol),
                          "objUid": objUid,
                          "objName": objName})
    return nodes

# Calculate and add the xy coordinates suitable for displaying the object nodes. The object nodes are distributed in a circle around the camera nodes.
def add_object_coords(object_nodes, poses_scaled_adjusted: dict, radius_extension) -> list:
    """
    In english:
    Args: object_nodes, poses_scaled_adjusted, radius_extension
    Returns: object nodes with xy coordinates
    Details: Object nodes are distributed in a circle around the camera nodes. 
    The object nodes are distributed on the enclosing circle of the camera nodes.
    """
    object_nodes_with_coords = []
    _, points = zip(*poses_scaled_adjusted.items())  # Extract the points from poses_scaled_adjusted
    # Get the enclosing circle of the camera nodes
    points = np.array(points).astype(np.int32)
    center, radius = cv2.minEnclosingCircle(points)
    # Extend the radius of the enclosing circle
    radius += radius_extension
    # Get the coordinates of the object nodes uniformly distributed on the enclosing circle
    for i, node in enumerate(object_nodes):
        angle = i * (2 * np.pi / len(object_nodes))
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        object_nodes_with_coords.append({**node, "x": round(x), "y": round(y)})
    return object_nodes_with_coords


# p-p edge
"""
"edges": [
    {
        "source": "cam0",  // source
        "target": "cam1"   // target
    },
]
"""
def get_pp_edges(pp_adj_list):
    edges = adjacency_list_to_edges(pp_adj_list)
    edges = [{"source": "cam" + str(source), "target": "cam" + str(target)} for source, targets in pp_adj_list.items() for target in targets]
    return edges

# p-o edge
"""
"edges": [
    {
        "source": "cam0",  // source
        "target": "obj0"   // target
    },
]
"""
def get_po_edges(po_adj_list):
    edges = []

    # adj_list to edges
    for source, targets in po_adj_list.items():
        for target in targets:
            edges.append((str(source), str(target)))
    # edges
    edges = [{"source": "cam" + str(source), "target": "obj" + str(target)} for source, targets in po_adj_list.items() for target in targets]
    return edges

def get_poses(sampled_frame_ids, traj_file):
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
            poses_from_traj[timestamp] = TrajStringToMatrix(line)[1][:, 3][:2].tolist()
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
    scale the poses to display and rounding
    Args:
        poses: a dictionary of poses
    Returns:
        a dictionary of scaled xy coordinates
    """
    # find the maximum abs value to determine the scaling factor
    max_abs_val = max(abs(value) for poses in poses.values() for value in poses)

    # find the scaling factor
    scaling_factor = SCALE_RATIO / max_abs_val

    # scale the poses
    scaled_points = {
        key: [round(val * scaling_factor) for val in values] for key, values in poses.items()
    }
    return scaled_points

def coord_adjustment(coordinates):
    """
    adjust the coordinates that are too close to each other
    Args:
        coordinates: a dictionary of coordinates
    Returns:
        a dictionary of adjusted coordinates
    """
    # for every 2 coordinates, if they are too close, adjust them in the line of the 2 coordinates
    # if the distance is less than DISTANCE_THRESHOLD, adjust the coordinates
    temp_coordinates = coordinates.copy()
    for key1, value1 in coordinates.items():
        for key2, value2 in coordinates.items():
            if key1 != key2:
                distance = np.linalg.norm(np.array(value1) - np.array(value2))
                if distance != 0 and distance < DISTANCE_THRESHOLD:
                    # adjust the coordinates
                    adjust_vector = (np.array(value2) - np.array(value1)) * (DISTANCE_THRESHOLD / distance) / 2
                    temp_coordinates[key1] = np.round((np.array(value1) - adjust_vector)).tolist()
                    temp_coordinates[key2] = np.round((np.array(value2) + adjust_vector)).tolist()
    return temp_coordinates



def main():
    n = len(sys.argv)
    if n != 3:
        print(f"Usage: python {sys.argv[0]} <path-to-data> <video-id>")
        sys.exit(1)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))    # switch to the directory of this script
    video_id = sys.argv[2]
    data_path = os.path.join(sys.argv[1], video_id)
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist.")
        sys.exit(1)

    # copy frames to img folder
    frames_path = os.path.join(data_path, f"{video_id}_frames", "lowres_wide")
    img_path = os.path.join(".", "img")
    if os.path.exists(img_path):
        shutil.rmtree(img_path)   # if the folder already exists, remove it
    os.makedirs(img_path, exist_ok=True)
    vid = re.compile(f"^{video_id}_")
    for _, _, files in os.walk(frames_path):
        for file in files:
            if not file.endswith(".png"):
                continue
            frame_id = vid.sub("", file).split(".png")[0]
            name_without_prefix = f"{round(float(frame_id), 3):.3f}.png"
            shutil.copyfile(os.path.join(frames_path, file), os.path.join(img_path, name_without_prefix))


    input_file = os.path.join(data_path, "refine_topo_gt.json")
    traj_file = os.path.join(data_path, f"{video_id}_frames", "lowres_wide.traj")

    with open(input_file, "r") as f:
        graph = json.load(f)

    # adj_list
    pp_adj_list = adjacency_matrix_to_list(graph["p-p"])
    po_adj_list = adjacency_matrix_to_list(graph["p-o"])
    # nodes
    poses_scaled_adjusted = get_camera_coords_scaled_adjusted(graph, traj_file)
    nodes = get_camera_nodes(graph, pp_adj_list, poses_scaled_adjusted)
    obj_nodes = get_object_nodes(graph)
    nodes += add_object_coords(obj_nodes, poses_scaled_adjusted, RADIUS_EXTENSION)
    # edges
    edges = get_pp_edges(pp_adj_list)
    edges += get_po_edges(po_adj_list)
    # output
    output_file = "graph_data.json"

    with open(output_file, "w") as f:
        json.dump({"nodes": nodes, "edges": edges,}, f, indent=4)

if __name__ == "__main__":
    main()