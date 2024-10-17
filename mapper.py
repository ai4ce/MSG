from typing import Tuple, List

import os
import json
import torch
from scipy.optimize import linear_sum_assignment
# from model import TopoMapper
# from arkit_dataset import AppleDataHandler, VideoDataset, arkit_collate_fn

import numpy as np


class TopoMapperHandler(object):

    def __init__(self, config, video_data_dir, video_id):
        self.config = config
        # self.mapper = TopoMapper(config)
        # map module
        self.video_id = video_id
        self.video_data_dir = video_data_dir
        gt_path = os.path.join(self.video_data_dir, 'refine_topo_gt.json')
        if os.path.exists(gt_path):
            self.gt = json.load(open(os.path.join(self.video_data_dir, 'refine_topo_gt.json')))
            self.frame_ids = self.gt['sampled_frames'] 
        else:
            self.frame_dir = os.path.join(self.video_data_dir, self.video_id+'_frames', 'lowres_wide')
            self.frame_ids = [fid.split(".png")[0].split("_")[-1] for fid in os.listdir(self.frame_dir) if fid.endswith(".png")]
        self.frame_ids.sort()
        self.num_frames = len(self.frame_ids)
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        # initialize the banks
        self.object_bank = None # object appearance bank
        self.object_feature_bank = None # object feature bank
        self.place_feature_bank = dict() # place feature bank
        # self.pp_adj = None
        self.pp_adj = torch.zeros((self.num_frames, self.num_frames))
        self.pp_threshold = config['pp_threshold']
        self.object_threshold = config['object_threshold']
        self.label2class = self.config['inv_class_map']
        

    def object_init(self, image_id, detections, object_embeddings):
        # bboxes: detection bounding boxes of a single image, note, not a batch
        # object_embeddings: embeddings of a single image
        # initialize the object bank, if empty, do nothing
        bboxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']
        uids = detections['uids']
        if bboxes.size(0) > 0 or object_embeddings.size(0)>0:
            self.object_bank = {} # stores object_id: (frame_id, bbox) and other info, like tracking
            # initialize the object feature bank

            object_feature_bank = list() # stores bank[object_id] = feature
            for bbox, obj_embed, label, score, uid in zip(bboxes, object_embeddings, labels, scores, uids):
                # new obj id
                obj_id = len(self.object_bank)
                # add to the object bank
                uniq_label = self.label2class[label.item()] + f":{int(uid)}"
                self.object_bank[obj_id] = {
                    'appearance': [(image_id, bbox, label, score, uniq_label)],
                    'class': label, # assign the class label to the object 
                    'uid': uid, # assign a unique id to the object -> from the groundtruth
                }
                # add to the object feature bank
                object_feature_bank.append(obj_embed)
            # convert to tensor
            self.object_feature_bank = torch.stack(object_feature_bank, dim=0) # M x Ho

    def obj_feature_update(self, object_id, object_embedding, weighted=True):
        # update the object feature bank with the new object embedding, weighed / unweighted average
        if weighted:
            count = len(self.object_bank[object_id]['appearance'])
            self.object_feature_bank[object_id] = (self.object_feature_bank[object_id] * count + object_embedding) / (count + 1)
        else:
            self.object_feature_bank[object_id] = (self.object_feature_bank[object_id] + object_embedding) / 2

    def gt_object_assignment(self, uids):
        query_ids = []
        existing_ids = []
        for i in range(len(uids)):
            uid = uids[i]
            for eid, ob in enumerate(self.object_bank):
                if self.object_bank[ob]['uid'] == uid:
                    query_ids.append(i)
                    existing_ids.append(eid)
                    break
        query_ids = np.array(query_ids)
        existing_ids = np.array(existing_ids)
        return query_ids, existing_ids

    def closest_object_assignment(self, sims):
        K, M = sims.size()
        query_ids = torch.arange(K).numpy()
        existing_ids = sims.argmax(dim=1).numpy()
        return query_ids, existing_ids

    def object_update(self, image_id, detections, object_embeddings):
        # check each object if it is a new object, if so enlarge the banks, if not append to the banks for update
        # compute consine similarity between the object embeddings and the object feature bank
        bboxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']
        uids = detections['uids']
        assert len(uids) == object_embeddings.size(0), (uids, object_embeddings.size(0), bboxes.size(), labels)
        if bboxes.size(0) == 0 or object_embeddings.size(0) == 0:
            return # skip empty detections
        
        # cosine similarity
        obj_sim = torch.cosine_similarity(object_embeddings.unsqueeze(1), self.object_feature_bank.unsqueeze(0), dim=-1)  # K x M
        # # dot product + sigmoid as similarity measure
        # obj_adj_dot = object_embeddings @ self.object_feature_bank.t()
        # obj_sim = torch.sigmoid(obj_adj_dot)
        
       
        # linear assignment matching
        query_ids, existing_ids = linear_sum_assignment(obj_sim.numpy(), maximize=True) # K x 2
        # print("============= matching ================")
        # print("hm query ids", query_ids)
        # print("hm existing ids", existing_ids)
        
        # closest matching
        # query_ids, existing_ids = self.closest_object_assignment(obj_sim)

        unmatched_obj_queries = set([i for i in range(object_embeddings.size(0))]) # a set of range(K)
        for i in range(len(query_ids)):
            matched_obj_id = existing_ids[i]
            query_id = query_ids[i] # shoud be the same as i

            if obj_sim[query_ids[i], matched_obj_id] > self.object_threshold:
                # print(f"at image {image_id}, new {query_ids[i]} matched with existing {matched_obj_id} with sim {obj_sim[query_ids[i], matched_obj_id]}")
                # remove from the unmatched obj queries
                unmatched_obj_queries.remove(query_id)
                # append to the object bank
                # if uid is included:
                uniq_label = self.label2class[labels[query_id].item()] + f":{int(uids[query_id])}"
                self.object_bank[matched_obj_id]['appearance'].append((image_id, bboxes[query_id], labels[query_id], scores[query_id], uniq_label))
                # update the object feature bank with the new object embedding, weighed / unweighted average
                self.obj_feature_update(matched_obj_id, object_embeddings[query_id], weighted=True)
        # handle the unmatched objects
        for i in list(unmatched_obj_queries):
            query_id = i
            # new object enlarge the object bank
            # print(f"new obj {query_id} is met, its gt uid is {uids[query_id]}, registered as {len(self.object_bank)}")
            uniq_label = self.label2class[labels[query_id].item()] + f":{int(uids[query_id])}"
            self.object_bank[len(self.object_bank)] = {
                'appearance': [(image_id, bboxes[query_id], labels[query_id], scores[query_id], uniq_label)],
                'class': labels[query_id], # assign the class label to the object 
                'uid': uids[query_id], # assign a unique id to the object -> like tracking
            }
            # enlarge the object feature bank
            self.object_feature_bank = torch.cat([self.object_feature_bank, object_embeddings[query_id].unsqueeze(0)], dim=0)


    def place_update(self, image_idx, place_embedding):
        # update the place feature bank
        self.place_feature_bank[image_idx] = place_embedding

    def get_pp(self,):
        # sort the place_feature bank according to the key (image id), organize the place feature bank into a matrix num_frames x embeddings:
        place_embeddings = []
        for image_id in sorted(self.place_feature_bank.keys()):
            place_embeddings.append(self.place_feature_bank[image_id])
        place_embeddings = torch.stack(place_embeddings, dim=0)
        # calculate the pairwise cosine similarity matrix
        self.pp_adj_sim = torch.cosine_similarity(place_embeddings.unsqueeze(1), place_embeddings.unsqueeze(0), dim=-1)
        
        # # calculate the pairwise dot product and sigmoid
        # pp_adj_dot = place_embeddings @ place_embeddings.t()
        # self.pp_adj_sim = torch.sigmoid(pp_adj_dot)
        # threshold the matrix
        self.pp_adj = (self.pp_adj_sim > self.pp_threshold).float()


    def adjust_pp(self,):
        # update pp_adj from pp_adj_sim by thresholding
        self.pp_adj = (self.pp_adj_sim > self.pp_threshold).float()


    def map_update(self, batch_data, batch_results):
        # first, organize the results as a list of batch_size, each element is a dict of image_id, bboxes, object_embeddings, place_embeddings
        list_results = []
        for i in range(batch_data['image_idx'].size(0)):
            list_results.append({
                'image_id': batch_data['image_idx'][i],
                'detections': batch_results['detections'][i],
                'object_embeddings': batch_results['embeddings'][i, :batch_results['detections'][i]['boxes'].size(0), :], # unpadding, get K x Ho
                'place_embeddings': batch_results['place_embeddings'][i], # Hp
                #'gt_bboxes'
                # 'object_labels': batch_data['object_labels'][i], # K
            })
        # then, iterate through the list, call object_init if the bank is empty,
        start_idx = 0
        # otherwise, update place feature bank, object bank, object feature bank
        for idx in range(start_idx, len(list_results)):
            # call map_update
            # print(">>>>>>>>>>>>>>>>>>update", idx)
            self.place_update(list_results[idx]['image_id'], list_results[idx]['place_embeddings'])
            if self.object_bank is None: # initialize object map
                self.object_init(list_results[idx]['image_id'], list_results[idx]['detections'], list_results[idx]['object_embeddings'])
            else:
                self.object_update(list_results[idx]['image_id'], list_results[idx]['detections'], list_results[idx]['object_embeddings'])

        
    def output_mapping(self, save_pp_sim=False, save_emb_dir=None):
        # out emb dir, added for api compatablity, left for future use
        # finish mapping, format the output as a json file for evaluation
        # get pp adj
        self.get_pp()
        # format the detections from object bank
        detections = {frame_id: dict() for frame_id in self.frame_ids}
        if self.object_bank is not None and len(self.object_bank) > 0:
            for obj_id, obj_dict in self.object_bank.items():
                for idx, bbox, label, score, uniq in obj_dict['appearance']:
                    frame_id = self.frame_ids[idx]
                    if frame_id not in detections:
                        raise ValueError("Frame id {} is not in the frame ids".format(frame_id))

                    if obj_id not in detections[frame_id]:
                        detections[frame_id][obj_id] = {
                            'bbox': bbox.tolist(), 
                            'label': label.item(),
                            'score': score.item(),
                            'uniq': uniq,
                            }
                        
                    # else: # NOTE: this should not happen
                        # raise ValueError("Object id {} is already in the detections".format(obj_id))

        mapping_results = {
            'video_id': self.video_id,
            'p-p': self.pp_adj.tolist(),
            'detections': detections,
        }
        # save place place cosine similarity for evaluation
        if save_pp_sim:
            mapping_results['pp-sim'] = self.pp_adj_sim.tolist()
        return mapping_results







## version 2: NOT USED. LEAVE FOR REFERENCE. COULD BE USED TO GET EMBEDDINGS FOR PLOTTING
## store all the object appearance, and do cluster ALL TOGEHTER in the end

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

class TopoMapperv2(object):

    def __init__(self, config, video_data_dir, video_id):
        self.config = config
        # self.mapper = TopoMapper(config)
        # map module
        self.video_id = video_id
        self.video_data_dir = video_data_dir
        gt_path = os.path.join(self.video_data_dir, 'refine_topo_gt.json')
        if os.path.exists(gt_path):
            self.gt = json.load(open(os.path.join(self.video_data_dir, 'refine_topo_gt.json')))
            self.frame_ids = self.gt['sampled_frames'] 
        else:
            self.frame_dir = os.path.join(self.video_data_dir, self.video_id+'_frames', 'lowres_wide')
            self.frame_ids = [fid.split(".png")[0].split("_")[-1] for fid in os.listdir(self.frame_dir) if fid.endswith(".png")]
        self.frame_ids.sort()
        self.num_frames = len(self.frame_ids)
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        # initialize the banks
        self.object_bank = list() # object appearance bank
        self.object_feature_bank = list() # object feature bank
        self.place_feature_bank = dict() # place feature bank
        # self.pp_adj = None
        self.pp_adj = torch.zeros((self.num_frames, self.num_frames))
        self.pp_threshold = config['pp_threshold']
        self.object_threshold = config['object_threshold']
        self.label2class = self.config['inv_class_map']
        

    def object_update(self, image_id, detections, object_embeddings):
        # bboxes: detection bounding boxes of a single image, note, not a batch
        # object_embeddings: embeddings of a single image
        # initialize the object bank, if empty, do nothing
        bboxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']
        uids = detections['uids']
        if bboxes.size(0) > 0 or object_embeddings.size(0)>0:
            for bbox, obj_embed, label, score, uid in zip(bboxes, object_embeddings, labels, scores, uids):
                # add to the object bank
                self.object_bank.append((image_id, bbox, label, score, uid))
                # add to the object feature bank
                self.object_feature_bank.append(obj_embed)


    def place_update(self, image_idx, place_embedding):
        # update the place feature bank
        self.place_feature_bank[image_idx] = place_embedding


    def get_pp(self,):
        # sort the place_feature bank according to the key (image id), organize the place feature bank into a matrix num_frames x embeddings:
        place_embeddings = []
        for image_id in sorted(self.place_feature_bank.keys()):
            place_embeddings.append(self.place_feature_bank[image_id])
        place_embeddings = torch.stack(place_embeddings, dim=0)
        # calculate the pairwise cosine similarity matrix
        self.pp_adj_sim = torch.cosine_similarity(place_embeddings.unsqueeze(1), place_embeddings.unsqueeze(0), dim=-1)
        # threshold the matrix
        self.pp_adj = (self.pp_adj_sim > self.pp_threshold).float()


    def adjust_pp(self,):
        # update pp_adj from pp_adj_sim by thresholding
        self.pp_adj = (self.pp_adj_sim > self.pp_threshold).float()


    def map_update(self, batch_data, batch_results):
        # first, organize the results as a list of batch_size, each element is a dict of image_id, bboxes, object_embeddings, place_embeddings
        list_results = []
        for i in range(batch_data['image_idx'].size(0)):
            list_results.append({
                'image_id': batch_data['image_idx'][i],
                'detections': batch_results['detections'][i], 
                'object_embeddings': batch_results['embeddings'][i, :batch_results['detections'][i]['boxes'].size(0), :], # unpadding, get K x Ho
                'place_embeddings': batch_results['place_embeddings'][i], # Hp
                #'gt_bboxes'
                # 'object_labels': batch_data['object_labels'][i], # K
            })
        # then, iterate through the list, call object_init if the bank is empty,
        start_idx = 0
        # otherwise, update place feature bank, object bank, object feature bank
        for idx in range(start_idx, len(list_results)):
            # call map_update
            self.place_update(list_results[idx]['image_id'], list_results[idx]['place_embeddings'])
            self.object_update(list_results[idx]['image_id'], list_results[idx]['detections'], list_results[idx]['object_embeddings'])
        
    
    def get_obj_cluster(self,):
        """
        Call clutering to get the object clusters (obj ids) from all the embeddings
        """
        if isinstance(self.object_feature_bank, list):
            obj_emb = torch.stack(self.object_feature_bank, dim=0).numpy()
        else:
            obj_emb = self.object_feature_bank.numpy()
        # cosine distance = 1 - cosine similarity
        distance_matrix = squareform(pdist(obj_emb, 'cosine'))
        db = DBSCAN(eps=0.6, min_samples=2, metric="precomputed").fit(distance_matrix)
        ids = db.labels_
        n_clusters_ = len(set(ids)) - (1 if -1 in ids else 0)
        n_noise_ = list(ids).count(-1)
        print(f"number of embeddings {ids.shape}, number of objects {n_clusters_}")
        print(f"number of noice points {n_noise_}, we treat them as alone objects each")
        obj_ids = ids.tolist()
        if n_noise_ > 0:
            alone_id = n_clusters_ # id starts from 0...
        for i in range(len(obj_ids)):
            if obj_ids[i] == -1:
                obj_ids[i] = alone_id
                alone_id += 1
        return obj_ids


    def output_obj_embedding(self, output_dir):
        """
        Output object embedding as a traj file and a embedding file..
        """
        # (N, h)
        os.makedirs(output_dir, exist_ok=True)
        object_bank = [(tp[0].item(), tp[1].tolist(), self.label2class[tp[2].item()], tp[3].item(), tp[4].item()) 
                       for tp in self.object_bank]
        with open(os.path.join(output_dir, 'appearances.json'), 'w') as f:
            json.dump(object_bank, f, indent=4)
        if isinstance(self.object_feature_bank, list):
           self.object_feature_bank = torch.stack(self.object_feature_bank, dim=0)
        np.save(os.path.join(output_dir, "object_embeddings.npy"), self.object_feature_bank.numpy())

        
    def output_mapping(self, save_emb_dir=None):
        # finish mapping, format the output as a json file for evaluation
        # get pp adj
        self.get_pp()
        # stack object embeddings
        print(f"total number of object embeddings {len(self.object_feature_bank)}")
        self.object_feature_bank = torch.stack(self.object_feature_bank, dim=0)
        if save_emb_dir is not None:
            self.output_obj_embedding(save_emb_dir)
        obj_ids = self.get_obj_cluster()
        # format the detections from object bank
        detections = {frame_id: dict() for frame_id in self.frame_ids}
        if self.object_bank is not None and len(self.object_bank) > 0:
            for obj_id, appr in zip(obj_ids, self.object_bank):
                idx, bbox, label, score, uniq = appr
                frame_id = self.frame_ids[idx]
                if frame_id not in detections:
                    raise ValueError("Frame id {} is not in the frame ids".format(frame_id))

                if obj_id not in detections[frame_id]:
                    detections[frame_id][obj_id] = list()
                detections[frame_id][obj_id].append({
                    'bbox': bbox.tolist(), 
                    'label': label.item(),
                    'score': score.item(),
                    'uniq': uniq.item(),
                })
                        
                # else: # NOTE: this should not happen
                    # raise ValueError("Object id {} is already in the detections".format(obj_id))

        # dump to json
        mapping_results = {
            'video_id': self.video_id,
            'p-p': self.pp_adj.tolist(),
            'detections': detections,
        }
        return mapping_results