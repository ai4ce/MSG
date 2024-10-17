# inference
# just obtain the results, no evaluation
# import yaml
import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import logging
from util.config_utils import get_configs
from util.transforms import get_transform
from util.box_utils import BBoxReScaler
from util.monitor import TrainingMonitor
from torch.utils.data import DataLoader
from arkit_dataset import AppleDataHandler 
from arkit_dataset import SimpleDataset, simple_collate_fn

from mapper import TopoMapperHandler as TopoMapper
# from mapper import TopoMapperv2 as TopoMapper

# from evaluator import Evaluator
from models.msg import MSGer
from util.checkpointing import load_checkpoint

import cv2
import supervision as sv
from typing import List

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

    labels = [
        "-".join(uniq.split(":")[::-1])
        for uniq in uniqs
    ]

    box_annotator = sv.BoxAnnotator()
    
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def visualize_det(out_path, det_type, pred, video_id, video_data_dir):
    frame_dir = os.path.join(video_data_dir, video_id+'_frames', 'lowres_wide')
    output_dir = os.path.join(out_path, "detection_frames_"+det_type)
    os.makedirs(output_dir, exist_ok=True)
    for frame_id, frame_dict in pred['detections'].items():
        image_path = os.path.join(frame_dir, f'{video_id}_{frame_id}.png')
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
            # mark with the unique id to show the instance results
            annotated_frame = annotate(image_source=image_source, boxes=bboxes, logits=logits, class_ids=labels, uniqs=uniqs)
            cv2.imwrite(os.path.join(output_dir, f'detected_{frame_id}.jpg'), annotated_frame)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(output_dir, output_file):
    """
    Create logger for traning records
    """
    logfile = output_file.split('.')[0]+"_inference.log"
    logpath = os.path.join(output_dir, logfile)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(logpath, mode='w')]
    )
    logger = logging.getLogger(__name__)
    return logger

# read frames sequentially, pass to the model, get the embeddings, and do the mapping
def inference_per_video(dataset, dataloader, config, mapper, model, device, backproc, logger):
    """
    Handles inference for each video
    Parameters:
     - dataset: the video dataset per video
     - dataloader: the torch dataloader
     - config: the config file
     - mapper: the mapping handler
     - model: the model used for evalaution, can be loaded from checkpoint or initialized from scratch
     - device: cuda:x or cpu
     - backproc: post processing handle for detection bounding boxes, used to back project them to the original image size.
    """
    model.eval()

    local_monitor = TrainingMonitor()
    local_monitor.add('running_loss_total')
    local_monitor.add('steps')

    with torch.no_grad():
        for batch in tqdm(dataloader):

            images = batch['image'].to(device)

            additional_info = {
                # 'gt_bbox': batch['bbox'].type(torch.FloatTensor).to(device),
                # 'obj_label': batch['obj_label'].to(device),
                # 'obj_idx': batch['obj_idx'].to(device),
                # 'mask': batch['mask'].to(device),
                # # 'image_idx': batch['image_idx']
                # # 'place_label': batch['place_label'].to(device),
            }
            
            if 'pred_bbox' in batch:
                additional_info['pred_bbox'] = batch['pred_bbox'].to(device)
                additional_info['pred_bbox_mask'] = batch['pred_bbox_mask'].to(device)
                additional_info['pred_label'] = batch['pred_label'].to(device)
                
            results = model(images, additional_info)


            # move the results to cpu
            results['place_embeddings'] = results['place_embeddings'].detach().cpu()
            # padded
            results['embeddings'] = results['embeddings'].detach().cpu()
            

            # rescale predicted bounding box to the  original image size
            results['detections'] = backproc.post_rescale_bbox(results['detections'])
            # pass the results to the mapper
            mapper.map_update(batch, results)

        # save the results


    # print(map_results)
    output_path = os.path.join(config['eval_output_dir'], config['eval_split'], dataset.video_id, "results.json")

    # obtain results
    # map_results = mapper.output_mapping()
    # # if save embedding, use topomapperv2 instead of topomapperhandler at line 19
    # map_results = mapper.output_mapping(save_emb_dir = os.path.dirname(output_path))
    # if save place-place similarity
    map_results = mapper.output_mapping(save_pp_sim=True)

    detector_type = config['detector']['model']
    if config['vis_det']:
        visualize_det(
            det_type=detector_type,
            pred=map_results,
            out_path = os.path.dirname(output_path),
            video_id = dataset.video_id,
            video_data_dir=dataset.video_data_dir,
        )

    # save the results
    if config["save_every"]:
        # check directory, make directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(map_results, f)

    return


def inference_map(config, logger):
    arkit_data = AppleDataHandler(config['dataset_path'], split=config['eval_split'])
    # print("Number of videos in the validation set: {}".format(len(arkit_data)))
    logger.info(f"Number of videos in the set: {len(arkit_data)}")
    # build or load model
    backproc = BBoxReScaler(orig_size=config['image_size'], new_size=config['model_image_size'], device='cpu')
    # get model
    device_no = config['device']
    device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
    model = MSGer(config, device)
    # if load from checkpoint
    if config["eval_chkpt"] is not None:
        chkpt_path = os.path.join(config["eval_output_dir"], "checkpoints", config["eval_chkpt"])
        load_checkpoint(
            model = model, 
            checkpoint_path=chkpt_path,
            logger = logger,
        )
        logger.info(f"Loading model from checkpoint: {chkpt_path}")
    else:
        assert config['associator']['model'] == 'SepMSG-direct', "No specified checkpoints for evaluation, so you can only do direct eval!"
    
    model = model.to(device)

    for i, next_video_id in enumerate(arkit_data.videos):
        # print("Processing video {}, progress {}/{}".format(next_video_id, i, len(arkit_data)))
        logger.info(f"Processing video {next_video_id}, progress {i+1}/{len(arkit_data)}")
        next_video_path = os.path.join(arkit_data.data_split_dir, next_video_id)
        dataset = SimpleDataset(arkit_data.data_split_dir, next_video_id, config, get_transform(config['model_image_size']), split=config['eval_split'])
        mapper = TopoMapper(config, next_video_path, next_video_id)
        dataloader = DataLoader(dataset, batch_size=config["eval_bs"], shuffle=False, num_workers=config["num_workers"], collate_fn=simple_collate_fn)
        inference_per_video(
            dataset, 
            dataloader, 
            config, 
            mapper, 
            model, 
            device, 
            backproc,
            logger,
        )

    return



if __name__ == '__main__':
    # get the config
    parser = argparse.ArgumentParser(description="Experiment configurations")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument('--experiment', type=str, help='Name of the experiment config to use')
    parser.add_argument('--output_dir', type=str, help='Output directory to save the results')
    parser.add_argument('--output_file', type=str, help='Output file name')
    parser.add_argument('--device', type=int, help="specify device")
    parser.add_argument('--eval_split', type=str, help= "specify evaluation split")
    parser.add_argument('--eval_output_dir', type=str, help="specify the eval model directory")
    parser.add_argument('--eval_chkpt', type=str, help="specify the checkpoint file")
    parser.add_argument('--vis_det', type=bool, help="if output frame visualization results")
    parser.add_argument('--object_threshold', type=float, help="specify object threshold if want")
    parser.add_argument('--pp_threshold', type=float, help="specify place threshold if want")
    
    args = parser.parse_args()

    base_config_dir = './configs/defaults'
    config = get_configs(base_config_dir, args, creat_subdir=False)

    eval_file = config["eval_split"]
    if config["eval_chkpt"] is not None:
        eval_file = config["eval_split"] + config["eval_chkpt"].split(".")[0]
    if config["detector"]["model"] == "grounding-dino":
        eval_file += "-gdino"
    logger = create_logger(config["eval_output_dir"], eval_file)
    # print(config)
    logger.info("Inference config: %s\n", json.dumps(config, indent=4))

    set_seed(config['seed'])
    # evaluate
    inference_map(config, logger)
    logger.info(f"Inference done!")
