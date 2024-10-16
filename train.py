# train
import argparse
import json
import os
import torch
import random
import numpy as np
import math
import logging
import wandb
from tqdm import tqdm

from util.config_utils import get_configs
from util.transforms import get_transform
from util.monitor import TrainingMonitor
from util.checkpointing import save_checkpoint, load_checkpoint, count_parameters
from torch.utils.data import DataLoader
from arkit_dataset import AppleDataHandler, VideoDataset, MultiVideoDataset, multivideo_collate_fn, arkit_collate_fn
from torch.optim.lr_scheduler import LambdaLR
from models.msg import MSGer

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_schedule_with_warmup(optimizer, type='cos', num_warmup_steps=2, num_training_steps=10000, last_epoch=-1):
    if type == 'linear':
        # linear scheduling
        def lr_func(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
    elif type == 'cos':
        lr_func = lambda current_step: min((current_step + 1) / (num_warmup_steps + 1e-8), 0.5 * (math.cos((current_step - num_warmup_steps) / num_training_steps * math.pi) + 1))
    elif type == 'warmup':
        # just warmup
        lr_func = lambda current_step: min((current_step + 1) / (num_warmup_steps + 1e-8), 1)
    else:
        # dont schedule
        lr_func = lambda current_step: 1

    return LambdaLR(optimizer, lr_func, last_epoch)


def create_logger(output_dir, output_file):
    """
    Create logger for traning records
    """
    logfile = output_file.split('.')[0]+".log"
    logpath = os.path.join(output_dir, logfile)
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(logpath, mode='w')]
    )
    logger = logging.getLogger(__name__)
    return logger
    


def train(config):
    logger = create_logger(output_dir=config["output_dir"], output_file=config["output_file"])

    if config["wandb"]:
        wandb.init(project="MSG", name=f"{config['experiment']}+{config['run_id']}", config=config)

    logger.info(f"Experiment directory created at {config['output_dir']}")
    logger.info("Training config: %s\n", json.dumps(config, indent=4))
    train_data = AppleDataHandler(config["dataset_path"], split=config['train_split'], video_batch_size=config['bs_video'])
    logger.info(f"Number of videos in the training set: {len(train_data)}")

    # build model and optimizer
    device_no = config['device']
    device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
    model = MSGer(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    
    scheduler = get_schedule_with_warmup(optimizer=optimizer,
                                         type=config['warmup'],
                                         num_warmup_steps=config['warmup_epochs'],
                                         num_training_steps=config['num_epochs'],
                                         )
    # if resume training
    if config['resume']:
        chkpt_path = config['resume_path']
        load_checkpoint(
            model= model,
            checkpoint_path=chkpt_path,
            optimizer=optimizer,
            logger = logger,
        )
        logger.info(f"Resume training by Loading model from checkpoint: {chkpt_path}")
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model and optimizer loaded, total parameter {total_params}, trainable {trainable_params}")
    logger.info(f"start training for {config['num_epochs']} epochs")
    
    monitor = TrainingMonitor()

    total_training_steps = 0
    training_steps_vid = 0

    if config['eval_step']> 0:
        model.eval()
        eval_res = eval(config, logger, model, device)
        if config["wandb"]:
            wandb.log(eval_res)
        model.train()

    for e in range(config['num_epochs']):
        model.train()
        train_data.shuffle()
        for i, next_video_id in enumerate(train_data):
            logger.info(f"Training on video {next_video_id}, progress {i+1}/{len(train_data)} Epoch {e+1}")
            
            dataset = MultiVideoDataset(video_data_dir = train_data.data_split_dir, 
                                        video_ids = next_video_id,
                                        configs = config,
                                        transforms = get_transform(config['model_image_size']),
                                        split="train", 
                                        batch_size=config["train_bs"]
                                        )
            dataloader = DataLoader(dataset, batch_size=config["train_bs"]//config["bs_video"], shuffle=True, 
                                    num_workers=config["num_workers"], collate_fn=multivideo_collate_fn)

            train_metric = train_per_video(
                model = model, 
                optimizer = optimizer, 
                # scheduler = scheduler,
                dataset = dataset,
                dataloader = dataloader,
                device = device,
                epoch = e,
                loss_params = config['loss_params'],
            )
            
            monitor.update(train_metric)
            total_training_steps += train_metric['training_steps']
            training_steps_vid += 1
            
            if training_steps_vid % config['log_every'] == 0 :
                monitor.get_avg()
                loggable = monitor.export_logging()
                logger.info(f"videos={training_steps_vid:07d}, steps={total_training_steps:07d}, {loggable}")
                if config["wandb"]:
                    wandb.log(monitor.export_wandb())
                    wandb.log({"learning_rate": optimizer.param_groups[0]['lr'],
                               "epoch": e})
                monitor.reset()

            # handles checkpointing, for later evaluation
            if training_steps_vid % config['chkpt_every'] == 0 and training_steps_vid > 0:
                model_cpu = model.cpu()
                save_path = os.path.join(config["output_dir"], "checkpoints", f"{e}-step{training_steps_vid}+.pth")
                logger.info(f"Saving checkpoints at {save_path}")
                save_checkpoint(model=model_cpu, optimizer=optimizer, checkpoint_path=save_path, config=config)
                model.to(device)

            if config['eval_step']> 0 and training_steps_vid % config['eval_step'] == 0:
                model.eval()
                eval_res = eval(config, logger, model, device)
                if config["wandb"]:
                    wandb.log(eval_res)
                model.train()
                

        # end of epoch, schedule learning rate
        scheduler.step()
    
        # save after each epoch
        model_cpu = model.cpu()
        save_path = os.path.join(config["output_dir"], "checkpoints", f"{e}-step{training_steps_vid}+.pth")
        logger.info(f"End of training, Saving checkpoints at {save_path}")
        save_checkpoint(model=model_cpu, optimizer=optimizer, checkpoint_path=save_path, config=config)
        model.to(device)
    wandb.finish()
    logger.info("Training Done!")
            

def train_per_video(model, optimizer, dataset, dataloader, device, epoch, loss_params):
    local_monitor = TrainingMonitor()
    local_monitor.add('running_loss_total')
    local_monitor.add('training_steps')
    
    for batch in dataloader:
        images = batch['image'].to(device)
        # potentially pass more information to the model
        additional_info = {
            'gt_bbox': batch['bbox'].type(torch.FloatTensor).to(device),
            'obj_label': batch['obj_label'].to(device),
            'obj_idx': batch['obj_idx'].to(device),
            'mask': batch['mask'].to(device)
        }
        place_labels = dataset.get_place_labels(batch['image_idx'], batch['num_per_vid'], batch['vid_idx'])
        results = model(images, additional_info)
        # spec
        total_loss, logs = model.compute_loss(
            results, 
            additional_info, 
            place_labels.type(torch.FloatTensor).to(device),
            weights = loss_params,
        )
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # handles logging
        local_monitor.add('running_loss_total', total_loss.item())
        local_monitor.add('training_steps', 1)
        local_monitor.update(logs)

    # prepare return
    local_metric = local_monitor.get_metric()
    return local_metric


# enable on the fly evaluation
from util.box_utils import BBoxReScaler
from mapper import TopoMapperHandler
from eval import eval_per_video

def eval(config, logger, model, device):
    arkit_data = AppleDataHandler(config['dataset_path'], split=config['eval_split'])
    logger.info(f"Evaluating: Number of videos in the validation set: {len(arkit_data)}")
    # build or load model
    backproc = BBoxReScaler(orig_size=config['image_size'], new_size=config['model_image_size'], device='cpu')
    
    model.eval()

    eval_results = dict()
    for i, next_video_id in enumerate(arkit_data.videos[:100]):
        # print("Processing video {}, progress {}/{}".format(next_video_id, i, len(arkit_data)))
        logger.info(f"Processing video {next_video_id}, progress {i+1}/{len(arkit_data)}")
        next_video_path = os.path.join(arkit_data.data_split_dir, next_video_id)
        dataset = VideoDataset(arkit_data.data_split_dir, next_video_id, config, get_transform(config['model_image_size']), split=config['eval_split'])
        mapper = TopoMapperHandler(config, next_video_path, next_video_id)
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
        eval_results[next_video_id] = eval_result_per_video

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
    logger.info(f"Evaluation done. Final Average avg pp: {avg_pp:.4f}, avg_po: {avg_po:.4f} avg_graph_iou: {avg_graph:.4f}")
    
    eval_results = {'avg_pp': avg_pp, 'avg_po': avg_po}
    return eval_results


if __name__ == '__main__':
    # get the config
    parser = argparse.ArgumentParser(description="Experiment configurations")
    parser.add_argument('--experiment', type=str, help='Name of the experiment config to use')
    parser.add_argument('--split', type=str, help='Name of the split to evaluate')
    parser.add_argument('--output_dir', type=str, help='Output directory to save the results')
    parser.add_argument('--output_file', type=str, help='Output file name')
    parser.add_argument('--learning_rate', type=float, help="learning rate")
    parser.add_argument('--num_epochs', type=int, help="number of training epochs in total")
    parser.add_argument('--warmup_epochs', type=int, help="number of epochs used for warmup")
    parser.add_argument('--warmup', type=str, help="types of scheduling")
    # loss hyper params
    parser.add_argument('--pr_loss', type=str, help='which loss for pr')
    parser.add_argument('--obj_loss', type=str, help='which loss for obj')
    # for focal loss
    parser.add_argument('--alpha', type=float, help='focal loss alpha')
    parser.add_argument('--gamma', type=float, help='focal loss gamma')
    # for bce loss
    parser.add_argument('--pos_weight', type=float, help='bce loss positive weights')
    parser.add_argument('--pp_weight', type=float, help='bce loss positive weights for place')
    # for infonce loss
    parser.add_argument('--temperature', type=float, help='infonce loss temperature')
    
    
    args = parser.parse_args()

    base_config_dir = './configs/defaults'
    config = get_configs(base_config_dir, args, creat_subdir=True) # for training, always create subdir
    # print(config)
    # fix seed
    set_seed(config["seed"])
    #train
    train(config)
