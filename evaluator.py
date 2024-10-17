# evaluator
# codes for evaluating the performance of the model
# input: groundtruth, prediction
# model will match the prediction to the groundtruth and output the metrics
# output: metrics
from typing import List
import numpy as np
from scipy.optimize import linear_sum_assignment

import supervision as sv

import warnings
warnings.filterwarnings("ignore", module="supervision.*")

import cv2
import torch
import os

from util.box_utils import compute_regular_iou, compute_ge_iou


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, class_ids: List[int], uniqs: List[str]) -> np.ndarray:
    padding = 20
    # Create a new image with padding
    annotated_frame = cv2.copyMakeBorder(image_source, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[1, 1, 1])

    xyxy = boxes.numpy()
    # padding the bounding box as well
    xyxy += padding
    class_ids = np.array(class_ids)
    cat_labels = []
    instance_ids = []
    pred_uids = []
    for uniq in uniqs:
        cat, uid, pred_uid = uniq.strip().split(":")
        instance_ids.append(int(uid))
        cat_labels.append(cat)
        pred_uids.append(pred_uid)
    instance_ids = np.array(instance_ids)
    cat_labels = np.array(cat_labels)
    detections = sv.Detections(xyxy=xyxy, class_id=class_ids, tracker_id=instance_ids, confidence=logits.numpy())

    # labels = [
    #     f"{phrase} p={logit:.2f}"
    #     for phrase, logit
    #     in zip(uniqs, logits)
    # ]
    labels = [
        "-".join(uniq.split(":")[::-1])
        for uniq in uniqs
    ]

    box_annotator = sv.BoxAnnotator()
    # annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    # box_annotator = sv.BoundingBoxAnnotator()
    
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def compute_iou(boxes_a, boxes_b):
    # for computing generalized iou
    iou = compute_ge_iou(boxes_a, boxes_b)
    # # for computing regular iou
    # iou = compute_regular_iou(boxes_a, boxes_b)
    return iou



class Evaluator(object):
    def __init__(self, video_data_dir, video_id, gt, pred, out_path, inv_class_map):
        self.out_path = out_path
        self.inv_class_map = inv_class_map
        self.gt = gt
        self.gt_obj2col = gt['obj2col']
        self.pred = pred
        self.gt_objs = []
        self.pred_objs = []
        self.frame2row = {frame_id: idx for idx, frame_id in enumerate(self.gt['sampled_frames'])}
        self.video_data_dir = video_data_dir
        self.video_id = video_id
        self.frame_dir = os.path.join(self.video_data_dir, self.video_id+'_frames', 'lowres_wide')

        self.matched_gt_indices = None
        self.get_gt_po()

    def get_gt_po(self,):
        num_frames = len(self.gt['sampled_frames'])
        num_obj = len(self.gt['obj2col'])
        po_adj = np.zeros((num_frames, num_obj), dtype=int)
        for frame_id in self.gt['sampled_frames']:
            if frame_id in self.gt['annotations']:
                anno = self.gt['annotations'][frame_id]
                for obj_id in anno:
                    po_adj[self.frame2row[frame_id], self.gt_obj2col[obj_id]] = 1
        self.gt['p-o'] = po_adj.tolist()

    
    # from annotations to object trajectories
    # annotations: {frame_id: {object_id: [x1, y1, x2, y2]}},
    # convert to {object_id: {frame_id: [x1, y1, x2, y2]}}
    def convert_gt_obj_traj(self):
        gt_traj = {}
        for frame_id, frame_dict in self.gt['annotations'].items():
            for obj_id in frame_dict:
                if obj_id not in gt_traj:
                    gt_traj[obj_id] = {}
                gt_traj[obj_id][frame_id] = frame_dict[obj_id] # object bounding box
        self.gt_obj_traj = gt_traj
        self.gt_objs = list(gt_traj.keys())
    
    # Note that the object id in the prediction is not the same as the groundtruth,
    # the evaluation is done by matching the object id in the prediction to the groundtruth first.
    def convert_pred_obj_traj(self):
        pred_traj = {}
        for frame_id, frame_dict in self.pred['detections'].items():
            for obj_id in frame_dict:
                if obj_id not in pred_traj:
                    pred_traj[obj_id] = {}
                if isinstance(frame_dict[obj_id], list):
                    appearance = frame_dict[obj_id][0] #NOTE if multiple same object in same frame, just take one
                else:
                    appearance = frame_dict[obj_id]
                pred_traj[obj_id][frame_id] = appearance['bbox']
        self.pred_obj_traj = pred_traj
        self.pred_objs = list(pred_traj.keys())

    def get_object_corres_score(self, iou_thres=0.0):
        # make a similarity matrix of size (num_gt_obj, num_pred_obj)
        num_gt_obj = len(self.gt_obj_traj)
        num_pred_obj = len(self.pred_obj_traj)
        # print(num_gt_obj, num_pred_obj)
        score_mat = np.zeros((num_gt_obj, num_pred_obj))

        # iterate through all object combinations, find the overlapping frame ids and calculate the corresponding ious
        for gt_idx, gt_obj_id in enumerate(self.gt_objs):
            for pred_idx, pred_obj_id in enumerate(self.pred_objs):
                gt_frame_ids = set(self.gt_obj_traj[gt_obj_id].keys())
                pred_frame_ids = set(self.pred_obj_traj[pred_obj_id].keys())
                overlap_frame_ids = gt_frame_ids.intersection(pred_frame_ids)
                #NOTE: there are chances that no overlapping frame ids, in this case, the iou is 0
                false_positive = len(pred_frame_ids.difference(gt_frame_ids))
                false_negative = len(gt_frame_ids.difference(pred_frame_ids))
                # get corresponding frame boxes and align in two matrices (N, 4) for iou calculation
                gt_boxes = np.zeros((len(overlap_frame_ids), 4))
                pred_boxes = np.zeros((len(overlap_frame_ids), 4))
                for idx, frame_id in enumerate(overlap_frame_ids):
                    gt_boxes[idx, :] = self.gt_obj_traj[gt_obj_id][frame_id]
                    pred_boxes[idx, :] = self.pred_obj_traj[pred_obj_id][frame_id]
                # calculate iou
                ious = compute_iou(gt_boxes, pred_boxes) #(N,)
                # ious is a numpy array of size (N,), each element is the iou of the corresponding frame
                # if iou is lower than the threshold, count it as mismatch
                # if iou is larger thant the threshold, count it as match and sum up the ious
                ious_match = ious[ious > iou_thres]
                num_match = len(ious_match)
                num_mismatch = len(ious) - num_match
                # get the similarity score
                # NOTE fp and np are included in the denominator
                # print(gt_obj_id, pred_obj_id, num_match, num_mismatch, np.sum(ious_match))
                score_mat[gt_idx, pred_idx] = np.sum(ious_match) / (num_match + num_mismatch + false_negative + false_positive) 
        
        self.score_mat = score_mat
        
    # matching
    def object_matching(self,):
        # hungarian matching
        row_ind, col_ind = linear_sum_assignment(1 - self.score_mat)
        self.row_ind = row_ind
        self.col_ind = col_ind
        # figure out 1-1 matching of object ids
        self.obj_match = {}
        self.pred_obj_numatched = set(self.pred_objs)
        for gt_idx, pred_idx in zip(row_ind, col_ind):
            self.obj_match[self.gt_objs[gt_idx]] = self.pred_objs[pred_idx]
            self.pred_obj_numatched.remove(self.pred_objs[pred_idx])

        # # for DEBUG, print
        # for gto, predo in self.obj_match.items():
        #     print(gto, self.gt['obj2col'][gto], predo)


    def print_matching(self,):
        for gt_obj_id, pred_obj_id in self.obj_match.items():
            print(gt_obj_id, pred_obj_id)


    def get_place_recall(self,):
        assert "pp-sim" in self.pred, (list(self.pred.keys()))
        recall = 0
        gt_pp_adj = np.array(self.gt['p-p']).astype(int)
        pp_sim = np.array(self.pred["pp-sim"], dtype=np.float32)
        dim = pp_sim.shape[0]
        sort_indx = np.argsort(pp_sim, axis=1)[:,::-1]
        for i in range(dim):
            for j in sort_indx[i][1:6]:
                if gt_pp_adj[i][j] == 1:
                    recall += 1
                break


        # eval recall based on the sorted index, mind the diagnal.
        recall = recall/dim
        return recall
        

    # visualize the detection results
    # det type: faster rcnn, grounding dino, gt
    def visualize_det(self, det_type):
        output_dir = os.path.join(self.out_path, "detection_frames_"+det_type)
        os.makedirs(output_dir, exist_ok=True)
        for frame_id, frame_dict in self.pred['detections'].items():
            image_path = os.path.join(self.frame_dir, f'{self.video_id}_{frame_id}.png')
            image_source = cv2.imread(image_path)
            # restore the classical detection data format
            bboxes = []
            labels = []
            logits = []
            uniqs = []
            for obj_id in frame_dict:
                bboxes.append(torch.tensor(frame_dict[obj_id]['bbox']))
                label = int(frame_dict[obj_id]['label'])
                # labels.append(self.inv_class_map[label] if label in self.inv_class_map else label)
                labels.append(label)
                logits.append(frame_dict[obj_id]['score'])
                uniqs.append(frame_dict[obj_id]['uniq']+f":p{obj_id}")
            if len(bboxes) == 0:
                cv2.imwrite(os.path.join(output_dir, f'detected_{frame_id}.jpg'), image_source)
            else:
                bboxes = torch.stack(bboxes)
                logits = torch.tensor(logits)
                # NOTE: mark with the unique id to show the instance results
                annotated_frame = annotate(image_source=image_source, boxes=bboxes, logits=logits, class_ids=labels, uniqs=uniqs)
                cv2.imwrite(os.path.join(output_dir, f'detected_{frame_id}.jpg'), annotated_frame)
        

    def get_object_adj(self,):
        # according to the object matches, compute the iou between 
        # the ground truth and predicted the adjacency matrices 
        # and calculate the matrix iou
        # first, build the adjacency matrix of the predicted object trajectories following the matched order
        num_obj = len(self.obj_match)
        num_frames = len(self.gt['sampled_frames'])
        # pred_adj = np.zeros((num_frames, num_obj))
        # append all unmatched pred object to the end
        num_unmatched_pred = len(self.pred_obj_numatched)
        # print("num matched", num_obj, "num unmatched", num_unmatched_pred)
        pred_adj = np.zeros((num_frames, num_obj+num_unmatched_pred))
        self.matched_gt_indices = []
        for idx, gt_obj_id in enumerate(self.obj_match):
            pred_obj_id = self.obj_match[gt_obj_id]
            for frame_id in self.pred_obj_traj[pred_obj_id]:
                pred_adj[self.frame2row[frame_id], idx] = 1
            self.matched_gt_indices.append(self.gt_obj2col[gt_obj_id])
        # for gt_obj_id, pred_obj_id in self.obj_match.items():
        #     for frame_id in self.pred_obj_traj[pred_obj_id]:
        #         pred_adj[self.frame2row[frame_id], self.gt_obj2col[gt_obj_id]] = 1
        # append all unmatched pred object to the end
        for exidx, pred_obj_id in enumerate(self.pred_obj_numatched):
            for frame_id in self.pred_obj_traj[pred_obj_id]:
                pred_adj[self.frame2row[frame_id], exidx+num_obj] = 1
        
        self.pred['p-o'] = pred_adj.tolist()


    def get_po_iou(self,):
        # calculate the iou of two binary matrices
        ori_gt_po_adj = np.array(self.gt['p-o']).astype(int)
        pred_po_adj = np.array(self.pred['p-o']).astype(int)

        # reorder gt_po_adj
        remaining_indices = [ i for i in range(ori_gt_po_adj.shape[1]) if i not in self.matched_gt_indices]
        reordered_indices = self.matched_gt_indices + remaining_indices
        gt_po_adj = ori_gt_po_adj[:, reordered_indices]

        # take the min, use extra 1's as penalizer
        min_col = np.minimum(gt_po_adj.shape[1], pred_po_adj.shape[1])
        # print("num objects, gt:", gt_po_adj.shape[1], "pred:", pred_po_adj.shape[1])
        # get intersection and union between the two [:min_col]
        self.po_intersection = np.sum(np.bitwise_and(gt_po_adj[:,:min_col], pred_po_adj[:,:min_col]))
        self.po_union = np.sum(np.bitwise_or(gt_po_adj[:,:min_col], pred_po_adj[:,:min_col]))
        # add residual 1's to the denominator as additinoal penalizer, note that one of the following two terms is 0
        self.po_residuals = np.sum(gt_po_adj[:, min_col:]) + np.sum(pred_po_adj[:, min_col:])
        # print("breakdown", self.po_intersection, self.po_union, self.po_residuals)
        self.po_iou = self.po_intersection / (self.po_union + self.po_residuals)

    def get_pp_iou(self,):
        gt_pp_adj = np.array(self.gt['p-p']).astype(int)
        pred_pp_adj = np.array(self.pred['p-p']).astype(int)
        # mask out diagnal 
        num_diag = gt_pp_adj.shape[0]
        np.fill_diagonal(gt_pp_adj, 1)
        np.fill_diagonal(pred_pp_adj, 1)
        # shape is already the same
        self.pp_intersection = np.sum(np.bitwise_and(gt_pp_adj, pred_pp_adj)) - num_diag
        self.pp_union = np.sum(np.bitwise_or(gt_pp_adj, pred_pp_adj)) - num_diag
        self.pp_iou = self.pp_intersection / self.pp_union

    # calculate graph IoU on the adjacency matrix.
    def get_graph_iou(self,):
        if self.pp_iou is None:
            self.get_pp_iou()
        if self.po_iou is None:
            self.get_po_iou()
        # iou of the entire place-object graph
        self.graph_iou = (self.pp_intersection+self.po_intersection) / (self.pp_union+self.po_union + self.po_residuals)

    def get_metrics(self,):
        self.convert_gt_obj_traj()
        self.convert_pred_obj_traj()
        self.get_object_corres_score()
        self.object_matching()
        self.get_object_adj()
        self.get_pp_iou()
        self.get_po_iou()
        self.get_graph_iou()
        # self.get_place_recall()
        return {
            'pp_iou': self.pp_iou,
            'po_iou': self.po_iou,
            'graph_iou': self.graph_iou,
            'num_pred_obj': len(self.pred_obj_traj),
            'num_unmatched_pred_obj': len(self.pred_obj_numatched), 
            'num_gt_obj': len(self.gt_obj2col),
            # 'place_recall': self.get_place_recall()
            
        }
