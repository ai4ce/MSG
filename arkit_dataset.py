# dataloader py
# this file contain the dataset class and corresponding dataloader of the apple arkit dataset, for MSG
# the data directory structure is as follows:
# validation/videoid(each is a video data directory)/ -> sub directory as follows:
# ./videoid_frames/lowres_wide/videoid_frameid.png -> contains the frames of the video

import os
import json
import numpy as np
import torch
# import cv2
import random
from torch.utils.data import Dataset

from torchvision.io import read_image
# from torchvision import tv_tensors
# from torchvision.transforms.v2 import functional as F
# from torchvision.tv_tensors import BoundingBoxes

from torch.nn.utils.rnn import pad_sequence

class AppleDataHandler:
    '''
    organize videos, provide next video(s) for training process
    '''
    def __init__(self, data_dir, split='train', video_batch_size=1):
        self.data_dir = data_dir
        self.split = split
        self.data_split_dir = os.path.join(self.data_dir, self.split)
        self.videos = sorted([f for f in os.listdir(self.data_split_dir) if os.path.isdir(os.path.join(self.data_split_dir, f))])
        self.num_videos = len(self.videos)
        self.vid_idx = 0 # pointer
        self.video_batch_size = video_batch_size
        

    def __len__(self,):
        return self.num_videos
    
    def __iter__(self,):
        return self
    
    def __next__(self,):
        if self.vid_idx >= self.num_videos:
            self.vid_idx = 0
            raise StopIteration
        else:
            ceil = min(self.vid_idx + self.video_batch_size, self.num_videos)
            current_video_batch = self.videos[self.vid_idx: ceil]
            self.vid_idx += self.video_batch_size
            return current_video_batch
    
    def shuffle(self,):
        # shuffle the videos for training
        random.shuffle(self.videos)

    def reset(self):
        self.vid_idx = 0
    



class VideoDataset(Dataset):

    def __init__(self, video_data_dir, video_id, configs, transforms, split="train"):
        self.split = split
        self.video_data_dir = os.path.join(video_data_dir, video_id)
        self.video_id = video_id
        self.transforms = transforms
        self.ori_image_size = configs['image_size']
        # get annotations
        self.gt = json.load(open(os.path.join(self.video_data_dir, 'refine_topo_gt.json'))) # 'topo_gt.json'
        # get the frame ids
        self.frame_dir = os.path.join(self.video_data_dir, self.video_id+'_frames', 'lowres_wide')
        #NOTE: NOT ALL frames are used -> [int(f.split('.')[0].split('_')[1]) for f in os.listdir(os.path.join(self.data_dir, self.video_id, 'videoid_frames', 'lowres_wide'))]
        self.frame_ids = self.gt['sampled_frames'] 
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)

        self.obj_id_offset = 0
        
        # get grounding dino detections if detector is grounding-dino
        self.use_gdino = False
        if configs["detector"]["model"] == "grounding-dino":
            self.use_gdino = True
            gdino_file = os.path.join(configs["detector"]["result_path"], split, video_id, 'eval_results.json')
            self.gdino_det = json.load(open(gdino_file))


        # other annotations
        self.obj2col = self.gt['obj2col'] # store which column in the gt annotation corresponds to which obj unique id
        self.pp_adj = torch.tensor(self.gt['p-p'])
        self.pp_adj.fill_diagonal_(1)
        # print("gt diag", self.pp_adj.diagonal())
        self.po_adj = torch.tensor(self.gt['p-o'])
        self.uid2obj = dict()
        for object_name in self.gt['uidmap']: # map object id to object name
            for object_id in self.gt['uidmap'][object_name]:
                self.uid2obj[object_id] = object_name
        self.class_map = configs['class_map'] # map object class name to class id, note that background should be set to 0
        self.image_size = configs['image_size']
        self.target_image_size = configs['model_image_size']
        self.new_width = self.target_image_size[1]
        self.new_height = self.target_image_size[0]
        self.orig_width = self.image_size[1]
        self.orig_height = self.image_size[0]

    def __len__(self,):
        return len(self.frame_ids)
    
    def get_det(self, frame_id):
        # read detection annotation, return list of bboxes and list of object ids
        bboxes = []
        obj_ids = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gt['annotations']: #NOTE: else this frame has no gt objct detections.
            det_dict = self.gt['annotations'][frame_id]
            for obj_id, bbox in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(bbox))
                obj_ids.append(self.obj2col[obj_id])
                obj_labels.append(self.class_map[self.uid2obj[obj_id]])
            # bboxes = tv_tensors.BoundingBoxes(bboxes, format='XYXY', canvas_size=self.image_size)
            # NOTE: for the older torchvision model BoundingBoxes are not there, gotta work around:
            bboxes = torch.stack(bboxes, dim=0)
        else:
            bboxes = torch.empty((0,4)) # no detection
        obj_ids = torch.as_tensor(obj_ids)
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_ids, obj_labels
    
    def get_pred_det(self, frame_id):
        bboxes = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gdino_det['detections']:
            det_dict = self.gdino_det['detections'][frame_id]
            for obj_id, det in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                obj_labels.append(det["label"])
            if len(bboxes)>0:
                bboxes = torch.stack(bboxes, dim=0)
            else:
                bboxes = torch.empty((0,4), dtype=torch.float)
        else:
            bboxes = torch.empty((0,4), dtype=torch.float) # no detection
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_labels
        
    
    def get_place_labels(self, frame_idxs):
        """
        index p-p adjencecy, find place labels, use for training place recognition
        Parameters:
         - frame_idxs: the frame idxs in this batch,
        Outputs:
         - place recognition labels, BxB binary matrice
        """
        place_labels = self.pp_adj[frame_idxs][:, frame_idxs]
        return place_labels
    
    
    def set_objidx_offset(self, offset):
        self.obj_id_offset = offset

    
    def __getitem__(self, idx):
        # get the frame id
        frame_id = self.frame_ids[idx]
        bboxes, obj_ids, obj_labels = self.get_det(frame_id) # (N, 4), (N) two lists
        
        data = dict()
        image_path = os.path.join(self.frame_dir, f'{self.video_id}_{frame_id}.png')
        image = read_image(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
            # transform boxes accordingly, surrogate when transform API doesn account for bboxes
            bboxes = bboxes.to(torch.float32)

            bboxes[:, 0::2] *= self.new_width / self.orig_width #(w, h, w ,h)
            bboxes[:, 1::2] *= self.new_height / self.orig_height
            bboxes = bboxes.to(torch.int64)
        # print("after transform", image1.size())
        # print("bbox after transform", bboxes1)
            
        data['image'] = image
        data['image_idx'] = torch.tensor(self.frame2idx[frame_id])

        data['bbox'] = bboxes
        data['obj_idx'] = obj_ids + self.obj_id_offset
        # data['place_label'] = place_label
        data['obj_label'] = obj_labels # this is class label
        
        if self.use_gdino:
            pred_box, pred_label = self.get_pred_det(frame_id)
            data['pred_bbox'] = pred_box

        # images: 2 x H x W x 3
        # detections: 2 x N x 4, N can be not the same
        # objects: 2 x N, N can be not the same
        # place_label: 1, if the two frames are the same place or not
        return data
    
class MultiVideoDataset(Dataset):
    """
    This class wraps VideoDataset.
    Supports multi video data loading. Only used for training
    """
    def __init__(self, video_data_dir, video_ids, configs, transforms, batch_size, split="train"):
        self.datasets = []

        objidx_offset_counter = 0
        for vid in video_ids:
            dt = VideoDataset(video_data_dir, vid, configs, transforms, split=split)
            dt.set_objidx_offset(objidx_offset_counter)
            self.datasets.append(dt)
            # accumulate the number of objects,
            # used to offset the object id so that they don't overlap
            objidx_offset_counter += len(dt.obj2col)

        self.video_ids = video_ids
        self.total_batch_size = batch_size
        self.bs_per_video = batch_size // len(video_ids)

        self.lens_per_video = torch.tensor([len(dt) for dt in self.datasets])
        self.min_len = self.lens_per_video.min().item()
        self.max_len = self.lens_per_video.max().item()

    def __len__(self,):
        return self.max_len
    
    def __getitem__(self, idx):
        batch_data = {}
        for dataset, length in zip(self.datasets, self.lens_per_video):
            index = idx % length
            batch_data[dataset.video_id] = dataset[index]
        return batch_data
    
    def get_place_labels(self, frame_idxs, num_per_vid, vid_idx):
        B = frame_idxs.size(0)
        place_labels = torch.zeros(B, B, dtype=torch.int)
        offset = 0
        for didx, dataset in enumerate(self.datasets):
            num_frames = num_per_vid[didx]
            assert torch.all(vid_idx[offset: offset + num_frames] == int(dataset.video_id))
            block_frame_idx = frame_idxs[offset: offset + num_frames]
            block_place_labels = dataset.get_place_labels(block_frame_idx)
            
            place_labels[offset:offset+num_frames, offset:offset+num_frames] = block_place_labels
            offset += num_frames
        return place_labels



def multivideo_collate_fn(batch):
    """
    1 x batch = bs x subdict 
    1 subdict = dict{videoId: data point per video}
    flatten the batch
    """
    # group by video id
    groups = {}
    for subdict in batch:
        for video_id, item in subdict.items():
            if video_id not in groups:
                groups[video_id] = list()
            groups[video_id].append(item)
    # flatten
    flat_batch = []
    num_per_vid = []
    batch_vid_idx = []
    for video_id in groups:
        bs_per_vid = len(groups[video_id])
        num_per_vid.append(bs_per_vid)
        batch_vid_idx.extend([int(video_id)]*bs_per_vid)
        flat_batch.extend(groups[video_id])

    # collate
    batch_images = torch.stack([item['image'] for item in flat_batch])
    batch_bboxes = pad_sequence([item['bbox'] for item in flat_batch], batch_first=True, padding_value=-1)
    batch_obj_ids = pad_sequence([item['obj_idx'] for item in flat_batch], batch_first=True, padding_value=-1)
    batch_obj_labels = pad_sequence([item['obj_label'] for item in flat_batch], batch_first=True, padding_value=-1)
    batch_mask = (batch_obj_ids != -1)
    batch_img_idx = torch.stack([item['image_idx'] for item in flat_batch])
    # # video info
    batch_vid_idx = torch.tensor(batch_vid_idx)
    batch_num_per_vid = torch.tensor(num_per_vid)

    return {
        'image': batch_images, # B x 3 x H x W
        'image_idx': batch_img_idx, # B
        'bbox': batch_bboxes, # B x padded N1 x 4
        'obj_idx': batch_obj_ids, # B x padded N1
        'obj_label': batch_obj_labels, # B x padded N1
        'mask': batch_mask, # B x padded N1 
        'vid_idx': batch_vid_idx, # (B,)
        'num_per_vid': batch_num_per_vid, # (num_videos,)
    }


    

    
def generate_mask(sequence, pad_value=0):
    return (sequence != pad_value).any(dim=-1) # if generates mask according to the bounding boxes (last dimension)


def arkit_collate_fn(batch):
    """
    custom collate function for the arkit dataset
    handles padding of the detection bounding boxes and other annotations with various lengths
    generate masks for the padded regions
    """
    # first images, no padding is needed:
    batch_images = torch.stack([item['image'] for item in batch])
    # then detections, padding is needed:
    batch_bboxes = pad_sequence([item['bbox'] for item in batch], batch_first=True, padding_value=-1)
    batch_obj_ids = pad_sequence([item['obj_idx'] for item in batch], batch_first=True, padding_value=-1)
    batch_obj_labels = pad_sequence([item['obj_label'] for item in batch], batch_first=True, padding_value=-1)
    # mask for padding
    batch_mask = (batch_obj_ids != -1)
    
    batch_img_idx = torch.stack([item['image_idx'] for item in batch])
    
    ret = {
        'image': batch_images, # B x 3 x H x W
        'image_idx': batch_img_idx, # B
        'bbox': batch_bboxes, # B x padded N1 x 4
        'obj_idx': batch_obj_ids, # B x padded N1
        'obj_label': batch_obj_labels, # B x padded N1
        'mask': batch_mask, # B x padded N1
    }
    
    if "pred_bbox" in batch[0]:
        batch_pred_bbox = pad_sequence([item['pred_bbox'] for item in batch], batch_first=True, padding_value=-1)
        batch_pred_bbox_mask = (batch_pred_bbox != -1).any(dim=2)
    
        ret['pred_bbox'] = batch_pred_bbox
        ret['pred_bbox_mask'] = batch_pred_bbox_mask
    
    return ret
    


######################################################
# for simple inference, when no ground truth
class SimpleDataset(Dataset):

    def __init__(self, video_data_dir, video_id, configs, transforms, split="train"):
        self.split = split
        self.video_data_dir = os.path.join(video_data_dir, video_id)
        self.video_id = video_id
        self.transforms = transforms
        self.ori_image_size = configs['image_size']
        # get the frame ids
        self.frame_dir = os.path.join(self.video_data_dir, self.video_id+'_frames', 'lowres_wide')
        self.frame_ids = [fid.split(".png")[0].split("_")[-1] for fid in os.listdir(self.frame_dir) if fid.endswith(".png")]
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)

        self.obj_id_offset = 0
        
        # get grounding dino detections if detector is grounding-dino
        self.use_gdino = False
        if configs["detector"]["model"] == "grounding-dino":
            if configs["detector"]["pre_saved"]:
                self.use_gdino = True
                gdino_file = os.path.join(configs["detector"]["result_path"], split, video_id, 'eval_results.json')
                self.gdino_det = json.load(open(gdino_file))


        self.class_map = configs['class_map'] # map object class name to class id, note that background should be set to 0
        self.image_size = configs['image_size']
        self.target_image_size = configs['model_image_size']
        self.new_width = self.target_image_size[1]
        self.new_height = self.target_image_size[0]
        self.orig_width = self.image_size[1]
        self.orig_height = self.image_size[0]

    def __len__(self,):
        return len(self.frame_ids)
    
    
    def get_pred_det(self, frame_id):
        bboxes = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gdino_det['detections']: 
            det_dict = self.gdino_det['detections'][frame_id]
            if isinstance(det_dict, dict):
                for obj_id, det in det_dict.items():
                    bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                    label = det["label"]
                    if isinstance(det["label"], str):
                        label = self.class_map[det["label"]]
                    obj_labels.append(label)
            else:
                for obj_id, det in enumerate(det_dict): # it is actually a list
                    bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                    label = det["label"]
                    if isinstance(det["label"], str):
                        label = self.class_map[det["label"]]
                    obj_labels.append(label)
            if len(bboxes)>0:
                bboxes = torch.stack(bboxes, dim=0)
            else:
                bboxes = torch.empty((0,4), dtype=torch.float)
        else:
            bboxes = torch.empty((0,4), dtype=torch.float) # no detection
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_labels
        
    
    
    def set_objidx_offset(self, offset):
        self.obj_id_offset = offset

    
    def __getitem__(self, idx):
        # get the frame id
        frame_id = self.frame_ids[idx]
        
        data = dict()
        image_path = os.path.join(self.frame_dir, f'{self.video_id}_{frame_id}.png')
        image = read_image(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
            
        data['image'] = image
        data['image_idx'] = torch.tensor(self.frame2idx[frame_id])
        
        if self.use_gdino:
            bboxes, pred_label = self.get_pred_det(frame_id)
            bboxes = bboxes.to(torch.float32)

            bboxes[:, 0::2] *= self.new_width / self.orig_width #(w, h, w ,h)
            bboxes[:, 1::2] *= self.new_height / self.orig_height
            # bboxes = bboxes.to(torch.int64)
            data['pred_bbox'] = bboxes
            data['pred_label'] = pred_label
            

        return data
    
    
def simple_collate_fn(batch):
    batch_images = torch.stack([item['image'] for item in batch])
    
    batch_img_idx = torch.stack([item['image_idx'] for item in batch])
    
    ret = {
        'image': batch_images, # B x 3 x H x W
        'image_idx': batch_img_idx, # B
    }
    
    if "pred_bbox" in batch[0]:
        batch_pred_bbox = pad_sequence([item['pred_bbox'] for item in batch], batch_first=True, padding_value=-1)
        batch_pred_bbox_mask = (batch_pred_bbox != -1).any(dim=2)
        batch_pred_label = pad_sequence([item['pred_label'] for item in batch], batch_first=True, padding_value=-1)
        
        ret['pred_bbox'] = batch_pred_bbox
        ret['pred_bbox_mask'] = batch_pred_bbox_mask
        ret['pred_label'] = batch_pred_label
        
    return ret
        