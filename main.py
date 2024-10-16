# main

import os
import json
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from arkit_config import arkit_config
from arkit_dataset import AppleDataHandler, VideoDataset, arkit_collate_fn
from mapper import TopoMapperHandler
from evaluator import Evaluator
from models.msg import MSGer

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms

def get_transform(size):
    transform_pipeline = transforms.Compose([
        # transforms.ToImage(),
        # transforms.ToDtype(torch.uint8, scale=True), # image is already in uint8 as a result of read_image()
        transforms.Resize(size=size, antialias=True), #NOTE: figure out the size from the model!
        transforms.ConvertImageDtype(torch.float32), #transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet default
    ])
    return transform_pipeline

class BBoxReScaler:
    def __init__(self, orig_size, new_size, device='cpu'):
        # ori size: original image size
        # new size: model image size
        self.orig_height, self.orig_width = orig_size
        self.new_height, self.new_width = new_size
        self.post_ratio_w = self.orig_width / self.new_width
        self.post_ratio_h = self.orig_height / self.new_height
        self.post_scaler = torch.tensor([self.post_ratio_w, self.post_ratio_h, self.post_ratio_w, self.post_ratio_h], device=device)
    
    def post_rescale_bbox(self, detections):
        for detection in detections:
            detection['boxes'] = (detection['boxes'].detach().to(self.post_scaler.device) * self.post_scaler).to(torch.int64)
            detection['labels'] = detection['labels'].detach().to(self.post_scaler.device)
            detection['scores'] = detection['scores'].detach().to(self.post_scaler.device)
        return detections


# # older version
# from old_detections import transforms
# # import torchvision.transforms as T
# def get_transform(size):
#     transform_pipeline = transforms.Compose([
#         transforms.ToDtype(torch.uint8, scale=True),
#         transforms.ScaleJitter(target_size=size, antialias=True),
#         transforms.ToDtype(torch.float32, scale=True),
#     ])
#     return transform_pipeline

# eval mode, read frames sequentially, pass to the model, get the embeddings, and do the mapping
def eval_per_video(next_video_path, next_video_id, arkit_config, split="val", feed="sequential"):
    dataset = VideoDataset(next_video_path, next_video_id, arkit_config, get_transform(arkit_config['model_image_size']), split=split, feed=feed)
    mapper = TopoMapperHandler(arkit_config, next_video_path, next_video_id)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=arkit_collate_fn)
    backproc = BBoxReScaler(orig_size=arkit_config['image_size'], new_size=arkit_config['model_image_size'], device='cpu')
    # get model
    device_no = arkit_config['device']
    device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
    model = MSGer(arkit_config, device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # print(batch)
            images1 = batch['image1'].to(device)
            images2 = batch['image2'].to(device)
            # potentially pass more information to the model
            additional_info1 = {
                'gt_bbox': batch['bbox1'].type(torch.FloatTensor).to(device),
                'obj_label': batch['obj_label1'].to(device),
                'obj_idx': batch['obj_idx1'].to(device),
                'mask': batch['mask1'].to(device)
                # 'place_label': batch['place_label'].to(device),
            }
            additional_info2 = {
                'gt_bbox': batch['bbox2'].type(torch.FloatTensor).to(device),
                'obj_label': batch['obj_label2'].to(device),
                'obj_idx': batch['obj_idx2'].to(device),
                'mask': batch['mask2'].to(device)
            }
            results = model(images1, images2, additional_info1, additional_info2)
            # move the results to cpu
            results['place_embeddings1'] = results['place_embeddings1'].detach().cpu()
            results['place_embeddings2'] = results['place_embeddings2'].detach().cpu()
            results['embeddings1'] = [emb.detach().cpu() for emb in results['embeddings1']]
            results['embeddings2'] = [emb.detach().cpu() for emb in results['embeddings2']]
            # rescale predicted bounding box to the  original image size
            # print("load", batch['bbox1'])
            # print("get", results['detections1'])
            results['detections1'] = backproc.post_rescale_bbox(results['detections1'])
            results['detections2'] = backproc.post_rescale_bbox(results['detections2'])
            # print(results)
            # print("scaled back", results['detections1'])
            # pass the results to the mapper
            mapper.map_update(batch, results)
            # break
        # save the results
    map_results = mapper.output_mapping()
    # print(map_results)
    output_path = os.path.join(arkit_config['output_dir'], split, next_video_id, arkit_config['output_file'])
    # check directory, make directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # evaluate the results
    evaluator = Evaluator(video_data_dir=next_video_path, 
                          video_id=next_video_id, 
                          gt=dataset.gt, 
                          pred=map_results, 
                          out_path=os.path.dirname(output_path),
                          inv_class_map = arkit_config['inv_class_map'],
                          )
    detector_type = arkit_config['detector']['model']
    if arkit_config['vis_det']:
        evaluator.visualize_det(det_type=detector_type)
    eval_results = evaluator.get_metrics()
    # combine the results and eval results for saving
    # print(eval_results)
    map_results['eval_metrics'] = eval_results
    # save the results
    with open(output_path, 'w') as f:
        json.dump(map_results, f)

    return eval_results


def eval_map(config):
    arkit_data = AppleDataHandler(config['dataset_path'], split=config['split'])
    print("Number of videos in the validation set: {}".format(len(arkit_data)))
    eval_results = dict()
    for i, next_video_id in enumerate(arkit_data.videos):
        print("Processing video {}, progress {}/{}".format(next_video_id, i, len(arkit_data)))
        next_video_path = os.path.join(arkit_data.data_split_dir, next_video_id)
        eval_result_per_video = eval_per_video(next_video_path, next_video_id, config, split=config['split'], feed="sequential")
        print(eval_result_per_video)
        eval_results[next_video_id] = eval_result_per_video
        # break

    return eval_results

# if main
if __name__ == '__main__':
    eval_results = eval_map(arkit_config)
    # get average eval results
    avg_pp = 0.
    avg_po = 0.
    avg_graph = 0.
    for vid, res in eval_results.items():
        avg_pp += res['pp_iou']
        avg_po += res['po_iou']
        avg_graph += res['graph_iou']
    avg_pp /= len(eval_results)
    avg_po /= len(eval_results)
    avg_graph /= len(eval_results)
    print("avg pp:", avg_pp, "avg_po:", avg_po, "avg_graph", avg_graph)
    # arkit_data = AppleDataHandler(arkit_config['dataset_path'], split="Validation")
    # next_video_path = os.path.join(arkit_config['dataset_path'], "Validation", next(iter(arkit_data.videos)))
    # next_video_id = next_video_path.split('/')[-1]
    # print(next_video_path)
    # dataset = VideoDataset(next_video_path, next_video_id, arkit_config, get_transform((224, 224)), split="Validation")
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=arkit_collate_fn)
    # for batch in dataloader:
    #     # pretty print the batch
    #     print(batch)
    #     break