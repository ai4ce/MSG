import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.transforms import v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.ops import roi_align
from torchvision.utils import save_image
from torchvision.models import resnet50
from models.matcher import HungarianMatcher
from models.encoders import Embedders
from models.associate import Asso_models

# from models.groundingdino_helper import GDino

from util.box_utils import enlarge_boxes, random_shift_boxes

class MSGer(nn.Module):
    def __init__(self, config, device):
        super(MSGer, self).__init__()
        device_no = config['device']
        self.device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
        classes = list(config['class_map'].keys())
        classes.remove('unknown')
        classes.remove('background')
        self.classes = classes
        self.text_class = " . ".join(self.classes)
        self.device = device

        # bounding box matcher
        self.box_matcher = HungarianMatcher()

        # object detection model
        self.detector = None
        self.detector_proc = None
        if config['detector']['model'] == 'fasterrcnn':
            self.build_fasterrcnn_detector(config['detector'], v2=False)
            self.get_detections = self.get_object_detections_fasterrcnn
        if config['detector']['model'] == 'fasterrcnnv2':
            self.build_fasterrcnn_detector(config['detector'], v2=True)
            self.get_detections = self.get_object_detections_fasterrcnn
        elif config['detector']['model'] == 'grounding-dino':
            self.build_groundingdino_detector(config['detector']) 
            self.get_detections = self.get_object_detections_groundingdino
        elif config['detector']['model'] == 'gt':
            self.detector = None
            self.detector_proc = None
            self.get_detections = self.get_gt_object_detections
        if config['detector']['freeze'] and self.detector is not None:
            for param in self.detector.parameters():
                param.requires_grad = False

        # feature extractor for objects
        self.obj_embedder = Embedders[config['obj_embedder']['model']](
            freeze = config['obj_embedder']['freeze'],
            weights = config['obj_embedder']['weights'],
            output_type = config['obj_embedder']['output_type']
        )

        # feature extractor for objects
        self.place_embedder = Embedders[config['place_embedder']['model']](
            freeze = config['place_embedder']['freeze'],
            weights = config['place_embedder']['weights'],
            output_type = config['place_embedder']['output_type']
        )
        
        # determine the object embedding type
        self.obj_embedder_type = None
        if 'direct' in config['associator']['model']:
            self.obj_embedder_type = 'direct'
        else:
            self.obj_embedder_type = 'feature'

        # update the loss keys to the associator
        loss_keys = ["pr_loss", "obj_loss", "temperature", "alpha", "gamma", "pos_weight", "pp_weight"]
        for k in loss_keys:
            if k in config:
                config['associator'][k] = config[k]
        self.association_model = Asso_models[config['associator']['model']](**config['associator'])

                
    # ------------------------------- detection -------------------------------------- #

    def build_fasterrcnn_detector(self, det_config, v2=False):
        if v2:
            self.detector = fasterrcnn_resnet50_fpn_v2(weights=det_config["weights"])
        else:
            self.detector = fasterrcnn_resnet50_fpn(weights=det_config["weights"])
        # return detector, None

    def get_object_detections_fasterrcnn(self, batch_images, info):
        list_images = list(image for image in batch_images)
        detections = self.detector(list_images)
        #additional filtering?
        # detections = self.filter_lowconf_detection(detections)
        return detections
    
    # NOTE USE PREVIOUSLY SAVED DETECTION RESULTS
    def build_groundingdino_detector(self, det_config):
        # self.detector = GDino(det_config["weights"], self.device, self.text_class)
        # if load saved detection results
        pass

    def get_object_detections_groundingdino(self, batch_images, info):
        # detections = self.detector(batch_images) # a wrapper for grounding dino prediction
        
        # NOTE USE PREVIOUSLY SAVED DETECTION RESULTS
        bs = batch_images.size(0)
        pred_bboxes = info.get("pred_bbox", torch.empty((bs,0,4), dtype=torch.float).to(self.device))
        pred_bbox_masks = info.get("pred_bbox_mask", torch.empty((bs,0), dtype=torch.bool).to(self.device))
        pred_labels = info.get("pred_label", torch.empty((bs,0), dtype=torch.int).to(self.device))
        detections = []
        for bboxes, masks, labels in zip(pred_bboxes, pred_bbox_masks, pred_labels):
            detections.append({
                'boxes': bboxes[masks],
                'labels': labels[masks], # do not involve label for now, can be easily added
                'uids': torch.ones(bboxes[masks].shape[0]),
                'scores': torch.ones(bboxes[masks].shape[0]),
            })
        
        return detections
    
    def filter_lowconf_detection(self, detections, conf_threshold=0.5):
        #filter out low confidence bounding boxes
        high_detections = []
        for detection in detections:
            # Get the bounding boxes, labels, and scores
            boxes = detection['boxes']
            labels = detection['labels']
            scores = detection['scores']

            # Filter out detections based on a confidence threshold (e.g., 0.5)
            high_confidence_idxs = scores > conf_threshold
            high_confidence_boxes = boxes[high_confidence_idxs]
            high_confidence_labels = labels[high_confidence_idxs]
            high_confidence_scores = scores[high_confidence_idxs]

            high_detections.append({
                'boxes': high_confidence_boxes,
                'labels': high_confidence_labels,
                'scores': high_confidence_scores
            })
        return high_detections
    
    def get_gt_object_detections(self, batch_images, info):
        # batch_bboxes: B x K x 4
        detections = convert_detections(info)
        return detections
    
    def get_box_match(self, detections, info):
        # match detected bounding boxes and targtes boxes
        targets = convert_detections(info)
        indices = self.box_matcher(detections, targets)
        return indices
    # ---------------------------------------------------------------------------- #


    # ------------------------------- embedding -------------------------------------- #
    
    def get_object_embeddings_from_feature(self, batch_features, detections):
        # Step 1: convert features (B, L, C) to (B, C, H, W)
        if len(batch_features.size()) == 3:
            h = w = int(batch_features.shape[1]**.5)
            assert h * w == batch_features.shape[1]
            x = batch_features.reshape(shape=(batch_features.shape[0], h, w, -1))
            x = torch.einsum('nhwc->nchw', x)
        else: # assue the shape is already (B, C, H, W)
            h, w = batch_features.size()[-2:]
            x = batch_features
        # spatial_scale = 1.0 / 14
        feature_dim = x.size(1)
        spatial_scale = h*1.0 / 224
        output_size = (1, 1)
        # Step 2: do the ROI align stuff
        num_detects = []
        batch_detects = 0
        list_boxes = []
        for det in detections:
            num_detects.append(det['boxes'].size(0))
            # list_boxes.append(enlarge_boxes(det['boxes'], scale=1.1))
            boxes = enlarge_boxes(det['boxes'], scale=1.1)
            if self.training:
                boxes = random_shift_boxes(boxes, shift_ratio=0.2)
            list_boxes.append(boxes)
            batch_detects += det['boxes'].size(0)
        if batch_detects > 0:
            all_embeddings = roi_align(x, list_boxes, output_size=output_size, spatial_scale=spatial_scale)
        else:
            all_embeddings = []
        # reorganize the crops to a list of crops
        embeddings_per_image = []
        start = 0
        for bid, count in enumerate(num_detects):
            if count == 0: 
                # handle empty detection
                embeddings_per_image.append(torch.empty(0,feature_dim).to(self.device))
            else:
                embeddings_per_image.append(all_embeddings[start:start+count].squeeze(-1).squeeze(-1))
            start += count

        return embeddings_per_image
        
    
    def get_object_embeddings(self, batch_images, detections):
        num_detects = []
        batch_detects = 0

        list_boxes = []
        for det in detections:
            num_detects.append(det['boxes'].size(0))
            # list_boxes.append(enlarge_boxes(det['boxes'], scale=1.1))
            boxes = enlarge_boxes(det['boxes'], scale=1.1)
            if self.training:
                boxes = random_shift_boxes(boxes, shift_ratio=0.3)
            list_boxes.append(boxes)
            batch_detects += det['boxes'].size(0)

        # to avoid empty input
        if batch_detects > 0: 
            all_crops = roi_align(batch_images, list_boxes, output_size=(224, 224))
            all_embeddings = self.obj_embedder(all_crops)
        else:
            all_embeddings = []

        # reorganize the crops to a list of crops
        embeddings_per_image = []
        start = 0
        for bid, count in enumerate(num_detects):
            if count == 0: 
                # handle empty detection
                embeddings_per_image.append(torch.empty(0,self.obj_embedder.feature_dim).to(self.device))
            else:
                embeddings_per_image.append(all_embeddings[start:start+count])
                # save_image(all_crops[start:start+count], f'all_crops_{bid}.png', nrow=4)
                # print(f'all_crops_{bid}.png', detections[bid]['uids'])
            start += count

        return embeddings_per_image

    def get_place_embeddings(self, batch_images):
        embeddings = self.place_embedder(batch_images)
        return embeddings
    
    # ---------------------------------------------------------------------------- #
    

    def compute_loss(self, results, additional_info, place_labels, weights):
        # box matching
        
        match_inds = self.get_box_match(results['detections'], additional_info)
        # print("match index", match_inds)
        
        total_loss, logs = self.association_model.get_loss(results, additional_info, match_inds, place_labels, weights)
        # object_loss, pr_loss = self.association_model.get_loss(results, association_sv, association_mask, place_labels)
        return total_loss, logs


    def forward(self, images, additional_info=None):
        # images = batch['image'] # B x 3 x H x W
        # object detection
        detections = self.get_detections(images, additional_info) # list of B elements, each contains a dict of detections of that image but in various lengths.
        # ------------------------------- embedding -------------------------------------- #
        
        # place embedding
        place_embeddings = self.get_place_embeddings(images) # B x Hp, dimension of place embeddings
        
        # object embedding
        # list of B elements, each is a tensor of embeddings (K x Ho) of that image but in various lengths K.
        if self.obj_embedder_type == 'direct':
            embeddings = self.get_object_embeddings(images, detections)
        elif self.obj_embedder_type == 'feature':
            if len(place_embeddings.size()) == 3: # transformer backbone
                embeddings = self.get_object_embeddings_from_feature(place_embeddings[:,1:,:], detections)
            else: # cnn backbone
                embeddings = self.get_object_embeddings_from_feature(place_embeddings, detections)
        else:
            raise NotImplementedError        
        # get association: object correspondence and place recognition

        # for decoder style msg
        boxes = [det['boxes'] for det in detections]
        results = self.association_model(embeddings, place_embeddings, boxes)

        results['detections'] = detections
        return results
    
    def compute_detection_loss(self, batch, results):
        det_loss = self.object_detection_loss(results['detections'], batch['bbox'], batch['object_labels'])
        
        return det_loss
    
    

def convert_detections(info):
    batch_bboxes = info['gt_bbox']
    batch_masks = info['mask']
    batch_labels = info['obj_label']
    batch_uids = info['obj_idx']
    # batch_masks: B x K, binary mask indicating whether the bbox is valid
    # format as if it is the output of the detector
    detections = []
    for bboxes, masks, labels, uids in zip(batch_bboxes, batch_masks, batch_labels, batch_uids):
        detections.append({
            'boxes': bboxes[masks],
            'labels': labels[masks],
            'uids': uids[masks],
            'scores': torch.ones(bboxes[masks].shape[0]),
        })
    return detections
