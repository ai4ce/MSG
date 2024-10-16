# evaluation script
import yaml
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
from arkit_dataset import AppleDataHandler, VideoDataset, arkit_collate_fn

from mapper import TopoMapperHandler as TopoMapper
# from mapper import TopoMapperv2 as TopoMapper

from evaluator import Evaluator
from models.msg import MSGer
from util.checkpointing import load_checkpoint

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
    logfile = output_file.split('.')[0]+"_eval.log"
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

# eval mode, read frames sequentially, pass to the model, get the embeddings, and do the mapping
def eval_per_video(dataset, dataloader, config, mapper, model, device, backproc, logger):
    """
    Handles evaluation for each video
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
    # print(local_monitor.metrics)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # print(batch)
            images = batch['image'].to(device)
            # potentially pass more information to the model
            additional_info = {
                'gt_bbox': batch['bbox'].type(torch.FloatTensor).to(device),
                'obj_label': batch['obj_label'].to(device),
                'obj_idx': batch['obj_idx'].to(device),
                'mask': batch['mask'].to(device),
                # 'image_idx': batch['image_idx']
                # 'place_label': batch['place_label'].to(device),
            }
            place_labels = dataset.get_place_labels(batch['image_idx']).to(device)
            
            if 'pred_bbox' in batch:
                additional_info['pred_bbox'] = batch['pred_bbox'].to(device)
                additional_info['pred_bbox_mask'] = batch['pred_bbox_mask'].to(device)
                
            results = model(images, additional_info)

            total_loss, logs = model.compute_loss(
                results, 
                additional_info, 
                place_labels.type(torch.FloatTensor).to(device),
                weights = config['loss_params'],
            )


            # move the results to cpu
            results['place_embeddings'] = results['place_embeddings'].detach().cpu()
            # padded
            results['embeddings'] = results['embeddings'].detach().cpu()
            
            # printss
            local_monitor.add('running_loss_total', total_loss.item())
            local_monitor.add('steps', 1)
            local_monitor.update(logs)

            # rescale predicted bounding box to the  original image size
            results['detections'] = backproc.post_rescale_bbox(results['detections'])
            # print("scaled back", results['detections1'])
            # pass the results to the mapper
            mapper.map_update(batch, results)

        # save the results
    # printss
    logger.info(json.dumps(local_monitor.get_avg(), indent=4))
    # print(local_monitor.avg_metrics)
    logger.info(local_monitor.export_logging())

    # print(map_results)
    output_path = os.path.join(config['eval_output_dir'], config['eval_split'], dataset.video_id, "eval_results.json")

    # obtain results
    # map_results = mapper.output_mapping()
    # # if save embedding
    # map_results = mapper.output_mapping(save_emb_dir = os.path.dirname(output_path))
    # if save place-place similarity
    map_results = mapper.output_mapping(save_pp_sim=True)
    # evaluate the results
    evaluator = Evaluator(video_data_dir=dataset.video_data_dir, 
                          video_id=dataset.video_id, 
                          gt=dataset.gt, 
                          pred=map_results, 
                          out_path=os.path.dirname(output_path),
                          inv_class_map = config['inv_class_map'],
                          )
    detector_type = config['detector']['model']
    if config['vis_det']:
        evaluator.visualize_det(det_type=detector_type)
    eval_results = evaluator.get_metrics()
    # combine the results and eval results for saving
    # print(eval_results)
    map_results['eval_metrics'] = eval_results
    # save the results
    if config["save_every"]:
        # check directory, make directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(map_results, f)

    return eval_results


def eval_map(config, logger):
    arkit_data = AppleDataHandler(config['dataset_path'], split=config['eval_split'])
    # print("Number of videos in the validation set: {}".format(len(arkit_data)))
    logger.info(f"Number of videos in the validation set: {len(arkit_data)}")
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
    eval_results = dict()
    for i, next_video_id in enumerate(arkit_data.videos):
        # print("Processing video {}, progress {}/{}".format(next_video_id, i, len(arkit_data)))
        logger.info(f"Processing video {next_video_id}, progress {i+1}/{len(arkit_data)}")
        next_video_path = os.path.join(arkit_data.data_split_dir, next_video_id)
        dataset = VideoDataset(arkit_data.data_split_dir, next_video_id, config, get_transform(config['model_image_size']), split=config['eval_split'])
        mapper = TopoMapper(config, next_video_path, next_video_id)
        dataloader = DataLoader(dataset, batch_size=config["eval_bs"], shuffle=False, num_workers=config["num_workers"], collate_fn=arkit_collate_fn)
        eval_result_per_video = eval_per_video(
            dataset, 
            dataloader, 
            config, 
            mapper, 
            model, 
            device, 
            backproc,
            logger,
        )
        # print(eval_result_per_video)
        logger.info("result per video: %s", json.dumps(eval_result_per_video, indent=4))
        eval_results[next_video_id] = eval_result_per_video
        # break

    return eval_results



if __name__ == '__main__':
    # get the config
    parser = argparse.ArgumentParser(description="Experiment configurations")
    parser.add_argument('--experiment', type=str, help='Name of the experiment config to use')
    parser.add_argument('--split', type=str, help='Name of the split to evaluate')
    parser.add_argument('--output_dir', type=str, help='Output directory to save the results')
    parser.add_argument('--output_file', type=str, help='Output file name')
    parser.add_argument('--device', type=int, help="specify device")
    parser.add_argument('--eval_split', type=str, help= "specify evaluation split")
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
    logger.info("Evaluation config: %s\n", json.dumps(config, indent=4))

    set_seed(config['seed'])
    # evaluate
    eval_results = eval_map(config, logger)
    # get average eval results
    avg_pp = 0.
    avg_po = 0.
    avg_graph = 0.
    # avg_recall = 0.
    for vid, res in eval_results.items():
        avg_pp += res['pp_iou']
        avg_po += res['po_iou']
        avg_graph += res['graph_iou']
        # avg_recall += res['place_recall']
    avg_pp /= len(eval_results)
    avg_po /= len(eval_results)
    avg_graph /= len(eval_results)
    # avg_recall /= len(eval_results)
    # print("avg pp:", avg_pp, "avg_po:", avg_po, "avg_graph", avg_graph)
    logger.info(f"Final Average avg pp: {avg_pp:.4f}, avg_po: {avg_po:.4f}, avg_graph_iou: {avg_graph:.4f}")
    # logger.info(f"Final Average avg pp: {avg_pp:.4f}, avg_po: {avg_po:.4f}, avg_graph_iou: {avg_graph:.4f}, avg_recall: {avg_recall:.4f}")
    logger.info(f"Evaluation done!")
