# utilities for using grounding-dino official
# make sure grounding dino is imported

# transform images on the fly
import torch
import torch.nn as nn
from groundingdino.util.inference import load_model
import torch.nn.functional as F
from torchvision.ops import box_convert
import bisect
from groundingdino.util.utils import get_phrases_from_posmap
from fuzzywuzzy import process

# disable the tokenizer warning!
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


MODEL_CONFIG_PATH = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

DEFAULT_WEIGHT_PATH = "./GroundingDINO/weights/groundingdino_swint_ogc.pth"

class GDino(nn.Module):
    def __init__(self, weight_path, device, text_class, input_size=(224, 224), output_size=800, max_size=1333):
        super(GDino, self).__init__()
        if weight_path == "DEFAULT":
            weight_path = DEFAULT_WEIGHT_PATH
        self.model = load_model(MODEL_CONFIG_PATH, weight_path)
        self.device = device
        # for rescaling image and boxes
        self.input_size = input_size
        self.size = self.get_size_with_aspect_ratio(input_size, size=output_size, max_size=max_size) # rescale size
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(self.size, input_size))
        ratio_width, ratio_height = ratios
        self.box_scaler = torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        ).to(self.device)
        source_h, source_w = input_size
        self.box_back_scaler = torch.Tensor([source_w, source_h, source_w, source_h]).to(device)

        self.classes = text_class.split(" . ")
        self.caption = self.preprocess_caption(text_class)
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        # tokenize 
        self.tokenizer = self.model.tokenizer
        self.tokenized = self.tokenizer(self.caption)
        self.sep_idx = [i for i in range(len(self.tokenized['input_ids'])) if self.tokenized['input_ids'][i] in [101, 102, 1012]]

    def preprocess_caption(self, caption):
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + " ."

    def get_size_with_aspect_ratio(self, image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)
    
    def on_device_resize_image(self, images):
        rescaled_image = F.interpolate(images, size=self.size, mode='bilinear', align_corners=False)
        # scaled_boxes = bboxes * self.box_scaler
        return rescaled_image

    def predict_image(self, images, captions, remove_combined=False):
        outputs = self.model(images, captions=captions)
        
        prediction_logits = outputs["pred_logits"].sigmoid()  # prediction_logits.shape = (bs, nq, 256)
        prediction_boxes = outputs["pred_boxes"]  # prediction_boxes.shape = (bs, nq, 4)
        # breakpoint()
        mask = prediction_logits.max(dim=2)[0] > self.box_threshold
        bs = mask.size(0)
        list_boxes = []
        list_logits = []
        for i in range(bs):
            list_boxes.append(prediction_boxes[i][mask[i]])
            list_logits.append(prediction_logits[i][mask[i]])
        # logits = prediction_logits[mask]  # logits.shape = (n, 256)
        # boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        list_phrases = []
        if remove_combined:
            
            for logits in list_logits:
                phrases = []
                for logit in logits:
                    max_idx = logit.argmax()
                    insert_idx = bisect.bisect_left(self.sep_idx, max_idx)
                    right_idx = self.sep_idx[insert_idx]
                    left_idx = self.sep_idx[insert_idx - 1]
                    phrases.append(get_phrases_from_posmap(logit > self.text_threshold, self.tokenized, self.tokenizer, left_idx, right_idx).replace('.', ''))
                list_phrases.append(phrases)
        else:
            for logits in list_logits:
                phrases = [
                    get_phrases_from_posmap(logit > self.text_threshold, self.tokenized, self.tokenizer).replace('.', '')
                    for logit
                    in logits
                ]
                list_phrases.append(phrases)
                
        list_confi_scores = [logits.max(dim=1)[0] for logits in list_logits]

        return list_boxes, list_confi_scores, list_phrases

    
    def forward(self, batch_images):
        bs = batch_images.size(0)
        captions = [self.caption] * bs
        detections = []
        rescaled_images = self.on_device_resize_image(batch_images)
        box_predictions, list_logits, list_phrases = self.predict_image(rescaled_images, captions)
        
        for boxes, logits, phrases in zip(box_predictions, list_logits, list_phrases):
            scale_back_boxes = self.post_process_result(boxes)
            class_ids = self.phrases2classes(phrases)
            assert scale_back_boxes.size(0) == class_ids.size(0), (scale_back_boxes.size(), class_ids.size(), phrases)
            # assert logits.size(0) == class_ids.size(0), (logits.size(), class_ids.size())
            detections.append(
                {
                    'boxes': scale_back_boxes,
                    'labels': class_ids,
                    'scores': logits,
                    'uids': class_ids,
                }
            )
        
        return detections
        

    
    def post_process_result(self, boxes):
        boxes = boxes * self.box_back_scaler
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        return xyxy

    def phrases2classes(self, phrases):
        class_ids = []
        for phrase in phrases:
            first_phrase = phrase.replace(" ", "")
            alt_phrase = phrase.replace(" ", "_") # account for the case that _ is used for space in category names
            found_match = False
            for class_ in self.classes:
                if class_ in first_phrase or class_ in alt_phrase:
                    class_ids.append(self.classes.index(class_)+1)
                    found_match = True
                    break
                elif first_phrase in class_ or alt_phrase in class_:
                    class_ids.append(self.classes.index(class_)+1)
                    found_match = True
                    break
            if not found_match:
                # use string matching library to find the closes match
                best_match = process.extractOne(phrase, self.classes)[0]
                class_ids.append(self.classes.index(best_match)+1)    
            
        return torch.Tensor(class_ids).to(self.device)

    
