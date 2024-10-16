# box utils
import numpy as np
import torch
from torchvision.ops.boxes import box_area

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
            for key in detection:
                if key == 'boxes':
                    detection[key] = (detection[key].detach().to(self.post_scaler.device) * self.post_scaler).to(torch.int64)
                else:
                    detection[key] = detection[key].detach().to(self.post_scaler.device)
        return detections

# from detr's code, for computing geIoU
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def compute_ge_iou(boxes_a, boxes_b):
    # a wrapper for the generalized_box_iou
    ta = torch.tensor(boxes_a)
    tb = torch.tensor(boxes_b)
    ge_iou = generalized_box_iou(ta, tb)
    return ge_iou.diag().numpy()


def compute_regular_iou(boxes_a, boxes_b):
    inter_xmin = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
    inter_ymin = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
    inter_xmax = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
    inter_ymax = np.minimum(boxes_a[:, 3], boxes_b[:, 3])

    inter_area = np.maximum(inter_xmax - inter_xmin, 0) * np.maximum(inter_ymax - inter_ymin, 0)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union_area = area_a + area_b - inter_area

    iou = inter_area / union_area

    return iou


def enlarge_boxes(boxes, image_size=(224, 224), scale=1.1):
    """
    Enlarge bounding boxes, and keep the enlarged box withint the image frame
    boxes are: tensor (N, 4), (x1, y1, x2, y2)
    image size: a tuple, here we use 224 x 224 since it has been preprocessed
    """
    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    box_sizes = boxes[:, 2:] - boxes[:, :2]
    new_box_sizes = box_sizes * scale

    new_boxes = torch.zeros_like(boxes)
    new_boxes[:, :2] = torch.max(box_centers - new_box_sizes / 2, torch.tensor([0, 0], dtype=boxes.dtype, device=boxes.device))
    new_boxes[:, 2:] = torch.min(box_centers + new_box_sizes / 2, torch.tensor(image_size, dtype=boxes.dtype, device=boxes.device))

    return new_boxes

def random_shift_boxes(bboxes, image_size = (224,224), shift_ratio=0.2):
    """
    random shift `shift_ratio` of a bound box
    boxes are: tensor (N, 4), (x1, y1, x2, y2)
    image size: a tuple, here we use 224 x 224 since it has been preprocessed
    """
    N = bboxes.size(0)
    if N == 0:
        return bboxes
    bbox_widths = bboxes[:, 2] - bboxes[:, 0]
    bbox_heights = bboxes[:, 3] - bboxes[:, 1]

    max_shifts_x = (bbox_widths * shift_ratio).int()
    max_shifts_y = (bbox_heights * shift_ratio).int()

    devc = bboxes.device

    shifts_x = torch.stack([torch.randint(-max_shift, max_shift + 1, (1,), device=devc) for max_shift in max_shifts_x])
    shifts_y = torch.stack([torch.randint(-max_shift, max_shift + 1, (1,), device=devc) for max_shift in max_shifts_y])

    bboxes[:, [0, 2]] += shifts_x  # x_min 和 x_max
    bboxes[:, [1, 3]] += shifts_y  # y_min 和 y_max

    width, height = image_size
    bboxes[:, [0, 2]] = torch.clamp(bboxes[:, [0, 2]], min=0, max=width)
    bboxes[:, [1, 3]] = torch.clamp(bboxes[:, [1, 3]], min=0, max=height)

    return bboxes



def get_box_coordinate(boxes, image_size=(224, 224)):
    """
    return the normalzied box center coordinate
    """
    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    normed_box_centers = box_centers * 1.0 / torch.tensor(image_size, dtype=boxes.dtype, device=boxes.device)

    return normed_box_centers

