# transform pipelines
import torch
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