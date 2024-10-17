# output embedding for localization
# localization: find the nearest neighbor on the graph
import os
import json
import torch
import torch.nn.functional as F
import argparse

from models.msg import MSGer
from torch.utils.data import DataLoader
from arkit_dataset import SimpleDataset, simple_collate_fn
from util.transforms import get_transform
from torchvision.io import read_image
from util.checkpointing import load_checkpoint
from util.config_utils import get_configs
import logging

class MSGLocalizer:
    """
    Leverage MSG for localization
    """
    def __init__(self, msg_path, model, data, data_path, device, image_transforms=None):
        """
        input:
            - msg_path: path to the scene MSG annotation (in the gt annotation format)
            - model: the model (checkpoint, loaded from outside)
            - frame_path: path to the frames
        """
        self.msg_path = msg_path
        self.model = model
        self.data_path = data_path
        self.device = device
        
        self.scene = self.convert_format_annotation(msg_path)
        self.frame_ids = self.scene['sampled_frames'] 
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        
        self.data = data
        self.image_transforms = image_transforms
        # obtain image embeddings for all images in the scene
        self.place_feature_bank = {}
        dataloader = DataLoader(self.data, batch_size=64, shuffle=False, num_workers=8, collate_fn=simple_collate_fn)
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                additional_info = {}
                
                # if there is object info
                # if 'pred_bbox' in batch:
                #     additional_info['pred_bbox'] = batch['pred_bbox'].to(device)
                #     additional_info['pred_bbox_mask'] = batch['pred_bbox_mask'].to(device)
                #     additional_info['pred_label'] = batch['pred_label'].to(device)
                
                results = self.model(images, additional_info)
                # move the results to cpu
                results['place_embeddings'] = results['place_embeddings'].detach().cpu()
                for i in range(batch['image_idx'].size(0)):
                    image_id = batch['image_idx'][i]
                    self.place_feature_bank[image_id] = results['place_embeddings'][i]
        place_embeddings = [self.place_feature_bank[image_id] for image_id in sorted(self.place_feature_bank.keys())]
        self.place_embeddings = torch.stack(place_embeddings, dim=0)
            
        
    def convert_format_annotation(self, msg_path):
        """
        convert the prediction file to annotation format
        input:
            - msg_pred: in prediction format
        output:
            - msg_anno: in gt annotation format
        """
        msg_pred = json.load(open(msg_path, 'r'))
        if "sampled_frames" in msg_pred: # the file is in annotation format
            return msg_pred
        else:
            video_id = msg_pred["video_id"]
            
            # get frame ids
            assert video_id == os.path.basename(self.data_path)
            with open(os.path.join(self.data_path, "refine_topo_gt.json"), 'r') as gf:
                gt = json.load(gf)
            msg_pred["sampled_frames"] = gt["sampled_frames"]
            # alternative: just access frame files for the frame id
            frame_path = os.path.join(self.data_path, f"{video_id}_frames", "lowres_wide")
            frames = [os.path.splitext(f)[0] for f in os.listdir(frame_path) if f.endswith((".png", ".jpg", ".jpeg"))]
            frame_ids = sorted([f.split("_")[1] for f in frames])
            msg_pred["sampled_frames"] = frame_ids
            
            # convert detections to annotation
            uidmap = {}
            annotations = {}
            for frame_id, det in msg_pred["detections"].items():
                if len(det) == 0:
                    continue
                annotations[frame_id] = dict()
                for obj_id, units in det.items():
                    bbox = units["bbox"]
                    annotations[frame_id][obj_id] = bbox
                    cat = units["uniq"].split(":")[0]
                    if cat not in uidmap:
                        uidmap[cat] = list()
                    uidmap[cat].append(obj_id)
            msg_pred["uidmap"] = uidmap
            msg_pred["annotations"] = annotations
            return msg_pred
            
                                                                                                                                                                                                                                                               
    
    def localize(self, image_path: str):
        """
        1. encode the image with model's place encoder
        2. find the nearest neighbor image on the MSG,hah
        3. return the frame id
        input: image_path
        output: the closest frame's id
        """
        query_image = read_image(image_path)
        if self.image_transforms is not None:
            query_image = self.image_transforms(query_image)
        query_image = query_image.unsqueeze(0).to(self.device)
        ret = self.model(query_image, {})
        query_embedding = ret["place_embeddings"].detach().cpu() # 1 x Hp
        # find the nearest neighbor
        cos_sim = F.cosine_similarity(query_embedding, self.place_embeddings, dim=1)
        closest_frame_idx = torch.argmax(cos_sim).item()
        closest_frame = self.frame_ids[closest_frame_idx]
        return closest_frame, cos_sim
        
        
        
    

def build_msg_localizer(msg_path, video_id, model_path=None, experiment_mode="localize", device=0, split="mini-val"):
    """
    Build MSGLocalizer instance
    1. load MSG file by video_id
    2. load MSG model
    3. initialize and return MSGLocalizer
    """
    # get configs
    base_config_dir = './configs/defaults'
    parser = argparse.ArgumentParser(description="Experiment configurations")
    parser.add_argument("--foo", default="construct args for compatibiliy")
    args = parser.parse_args([])
    args.__dict__.update({
        "experiment": experiment_mode,
        "device": device,
        "eval_split": split,
    })
    config = get_configs(base_config_dir, args, creat_subdir=False)
    
    # get model
    device_no = config['device']
    device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
    model = MSGer(config, device)
    
    # load from checkpoint
    if model_path is None:
        if 'eval_output_dir' not in config or config['eval_output_dir'] is None:
            raise AttributeError
        else:
            model_path = config['eval_output_dir']
    assert config["eval_chkpt"] is not None
    chkpt_path = os.path.join(model_path, "checkpoints", config["eval_chkpt"])
    logger = logging.getLogger(__name__)
    load_checkpoint(
        model = model, 
        checkpoint_path=chkpt_path,
        logger = logger,
    )
    logger.info(f"Loading model from checkpoint: {chkpt_path}")
    model = model.to(device)
    model.eval()

    # prepare data
    data_split_dir = os.path.join(config["dataset_path"], config["eval_split"])
    video_path = os.path.join(data_split_dir, video_id)
    transforms = get_transform(config['model_image_size'])
    dataset = SimpleDataset(data_split_dir, video_id, config, transforms, split=config['eval_split'])
    
    # initialize
    localizer = MSGLocalizer(
        msg_path = msg_path,
        model = model,
        data = dataset,
        data_path = video_path,
        device = device,
        image_transforms = transforms,
    )
    
    return localizer


        
    


if __name__ == '__main__':
    # show use case of MSGLocalizer
    # from localization import build_msg_localizer
    video_id = "41069042"
    predicted_msg_file = "./exp-results/aomsg/LOG_DATE/Test/41069042/eval_results.json"
    localizer = build_msg_localizer(
        msg_path = predicted_msg_file,
        video_id = video_id,
        experiment_mode="localize",
        device = 0,
        split = "mini-val",
    )
    
    # # single check
    # query_image_path = "./data/msg/mini-val/41069042/41069042_frames/lowres_wide/41069042_3048.737.png"
    # loc, sim = localizer.localize( query_image_path)
    # print(loc)
    # print(sim.size(), len(localizer.frame2idx), len(localizer.frame_ids))
    # print(localizer.frame2idx[loc], sim.argmax(), sim[sim.argmax()])
    # trueid = localizer.frame2idx["3048.737"]
    # print(trueid, sim[trueid])
    
    # recall check
    allquerys = [f for f in os.listdir(f"./data/msg/mini-val/{video_id}/{video_id}_frames/lowres_wide/") if f.endswith(".png")]
    recall = 0.
    for q in allquerys:
        frame_id = os.path.splitext(q)[0].split("_")[1]
        query_image_path = os.path.join(f"./data/msg/mini-val/{video_id}/{video_id}_frames/lowres_wide/", q)
        loc, _ = localizer.localize(query_image_path)
        if loc == frame_id:
            recall += 1
    print("recall:", recall / len(allquerys))
    
    