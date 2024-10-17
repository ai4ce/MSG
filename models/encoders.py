# object encoding modules
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torchvision.models import resnet50, resnet18, convnext_tiny, convnext_small, convnext_base
# from transformers import AutoImageProcessor, ConvNextModel
from transformers import Dinov2Model, ViTModel


class ResNetEmbedder(nn.Module):
    def __init__(self, model_type, output_type='vec', freeze=True, weights="DEFAULT"):
        super().__init__()
        self.backbone = None
        if model_type == 'resnet50':
            self.backbone = resnet50(weights=weights)
        elif model_type == 'resnet18':
            self.backbone = resnet18(weights=weights)
        else:
            raise NotImplementedError
        
        self.feature_dim = int(self.backbone.fc.in_features)
        
        if output_type == 'vec':
            self.backbone.fc = nn.Identity()
        elif output_type == 'feature':
            # Remove the avgpool and fc layer to output the feature map
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-3])
        else:
            raise NotImplementedError

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward(self, pixel_values):
        return self.backbone(pixel_values)
    

class ConvNextEmbedder(nn.Module):
    def __init__(self, model_type, output_type='vec', freeze=True, weights='DEFAULT'):
        super().__init__()
        if model_type == "convnext-tiny-224":
            # this bare model from hf just outputs the featuremap
            # self.backbone = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
            self.backbone = convnext_tiny(weights=weights)
        elif model_type == "convnext-small-224":
            self.backbone = convnext_small(weights=weights)
        elif model_type == "convnext-base-224":
            self.backbone = convnext_base(weights=weights)
        else:
            raise NotImplementedError
        
        self.feature_dim = int(self.backbone.classifier[2].in_features)

        if output_type == "vec":
            #this is applied after the avgpool: AdaptiveAvgPoo2d(output_size=1x1)
            self.backbone.classifier[2] = nn.Identity()
        elif output_type == "feature":
            # Modify the model to remove the avgpool and classifier for feature map output
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            # Remove classifier
            self.backbone[-1] = nn.Sequential(*list(self.backbone[-1].children())[:-1]) 
        else:
            raise NotImplementedError 

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # for param in self.backbone.classifier[2].parameters():
            #     param.requires_grad = True
            self.backbone.eval()

    def forward(self, pixel_values):
        return self.backbone(pixel_values)
    

class ViTEmbedder(nn.Module):
    # Output type: cls, mean of the tokens, max, or perhaps tokens
    def __init__(self, model_type, output_type='mean', freeze=True,weights="DEFAULT"):
        super().__init__()
        if model_type == "vit_base":
            self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        elif model_type == "vit_large":
            self.vit_model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        elif model_type == "vit_huge":
            self.vit_model = ViTModel.from_pretrained("google/vit-huge-patch14-224")
        else:
            raise NotImplementedError(f"Model type {model_type} is not implemented")

        if freeze:
            for param in self.vit_model.parameters():
                param.requires_grad = False
            self.vit_model.eval()
        
        self.feature_dim = self.vit_model.config.hidden_size
        self.adaptor = nn.Identity()

        if output_type == 'mean':
            print("Using mean pooling")
            self.projection = self.mean_projection
        elif output_type == 'cls':
            self.projection = self.cls_projection
        elif output_type == 'feature':
            self.projection = self.seq_projection
        elif output_type == 'max':
            print("Using max pooling")
            self.projection = self.max_projection
        elif output_type.startswith('gem'):  # For example: gem_3
            print(f"Using Generalized Mean Pooling with p={output_type.split('_')[1]}")
            p = float(output_type.split('_')[1])
            self.projection = self.gem_projection
            self.p = nn.Parameter(torch.ones(1) * p)
            if freeze:
                self.p.requires_grad = False
        else:
            raise NotImplementedError(f"Output type {output_type} is not implemented")

    def mean_projection(self, x):
        return x[:, 1:, :].mean(dim=1)
    
    def max_projection(self, x):
        x_bhl = x.permute(0, 2, 1)
        xp_bh = F.adaptive_max_pool1d(x_bhl, output_size=1).squeeze(2)
        return xp_bh
    
    def gem_projection(self, x, eps=1e-6):
        x_clamped = F.relu(x).clamp(min=eps)
        gem_pooled = (x_clamped.pow(self.p).mean(dim=1, keepdim=False)).pow(1./self.p)
        return gem_pooled
    
    def cls_projection(self, x):
        return x[:, 0, :]
    
    def seq_projection(self, x):
        return x  # B x L

    def forward(self, pixel_values):
        outputs = self.vit_model(pixel_values=pixel_values)
        last_hidden_states = outputs.last_hidden_state  # B x L x h
        embs = self.projection(last_hidden_states)
        embs = self.adaptor(embs)
        return embs



class DINOv2Embedder(nn.Module):
    # output type: cls or mean of the tokens, or perhaps tokens?
    def __init__(self, model_type, output_type='mean', freeze=True, weights="DEFAULT"):
        super().__init__()
        if model_type == "dinov2_vits14":
            # dino_image_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.dino_model = Dinov2Model.from_pretrained("facebook/dinov2-small")
        elif model_type == "dinov2_vitb14":
            self.dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        elif model_type == "dinov2_vitl14":
            self.dino_model = Dinov2Model.from_pretrained("facebook/dinov2-large")
        # elif model_type == "dinov2_vitg14":
        #     self.dino_model = Dinov2Model.from_pretrained("facebook/dinov2-huge")
        else:
            raise NotImplementedError(model_type)

        if freeze:
            for param in self.dino_model.parameters():
                param.requires_grad = False
            self.dino_model.eval()
        
        self.feature_dim = self.dino_model.encoder.layer[-1].mlp.fc2.out_features
        self.adaptor = nn.Identity()

        if output_type == 'mean':
            print("use mean")
            self.projection = self.mean_projection
        elif output_type == 'cls':
            self.projection = self.cls_projection
        elif output_type == 'feature':
            self.projection = self.seq_projection
        elif output_type == 'max':
            print("use max")
            self.projection = self.max_projection
        elif output_type.startswith('gem'): # for example: gem_3
            print("use", output_type)
            p = float(int(output_type.split('_')[1]))
            self.projection = self.gem_projection
            self.p = nn.Parameter(torch.ones(1) * p)
            if freeze:
                self.p.requires_grad = False
        else:
            raise NotImplementedError

    def mean_projection(self, x):
        return x[:, 1:, :].mean(dim=1)
    
    def max_projection(self, x):
        x_bhl = x.permute(0, 2, 1)
        xp_bh = F.adaptive_max_pool1d(x_bhl, output_size=1).squeeze(2)
        return xp_bh
    
    def gem_projection(self, x, eps=1e-6):
        x_clamped = F.relu(x).clamp(min=eps)
        gem_pooled = (x_clamped.pow(self.p).mean(dim=1, keepdim=False)).pow(1./self.p)
        return gem_pooled        
    
    def cls_projection(self, x):
        return x[:, 0, :]
    
    def seq_projection(self, x):
        return x # B x L

    def forward(self, pixel_values):
        inputs = {'pixel_values': pixel_values}
        outputs = self.dino_model(**inputs)
        last_hidden_states = outputs.last_hidden_state # B x L x h
        # print("dino last hidden states", last_hidden_states.size())
        embs = self.projection(last_hidden_states)
        # print("dino emb", embs.size())
        # embs = last_hidden_states[:, 1:, :].mean(dim=1) # B x h, NOTE: its an average over all image tokens
        embs = self.adaptor(embs)
        return embs        



            


def ResNet50_Embedder(**kwargs):
    return ResNetEmbedder(model_type='resnet50', **kwargs)

def ResNet18_Embedder(**kwargs):
    return ResNetEmbedder(model_type='resnet18', **kwargs)

def ConvNextTiny_Embedder(**kwargs):
    return ConvNextEmbedder(model_type='convnext-tiny-224', **kwargs)

def ConvNextSmall_Embedder(**kwargs):
    return ConvNextEmbedder(model_type='convnext-small-224', **kwargs)

def ConvNextBase_Embedder(**kwargs):
    return ConvNextEmbedder(model_type='convnext-base-224', **kwargs)

def DINOv2Small_Embedder(**kwargs):
    return DINOv2Embedder(model_type="dinov2_vits14", **kwargs)

def DINOv2Base_Embedder(**kwargs):
    return DINOv2Embedder(model_type="dinov2_vitb14", **kwargs)

def DINOv2Large_Embedder(**kwargs):
    return DINOv2Embedder(model_type="dinov2_vitl14", **kwargs)
    
def ViTBase_Embedder(**kwargs):
    return ViTEmbedder(model_type="vit_base", **kwargs)

def ViTLarge_Embedder(**kwargs):
    return ViTEmbedder(model_type="vit_large", **kwargs)

def ViTHuge_Embedder(**kwargs):
    return ViTEmbedder(model_type="vit_huge", **kwargs)



# def DINOv2Huge_Embedder(**kwargs):
#     return DINOv2Embedder(model_type="dinov2_vitg14", **kwargs)


Embedders = {
    'resnet50': ResNet50_Embedder,
    'resnet18': ResNet18_Embedder,
    'convnext-tiny': ConvNextTiny_Embedder,
    'convnext-small': ConvNextSmall_Embedder,
    'convnext-base': ConvNextBase_Embedder,
    'dinov2-small': DINOv2Small_Embedder,
    'dinov2-base': DINOv2Base_Embedder,
    'dinov2-large': DINOv2Large_Embedder,
    'vit-base': ViTBase_Embedder,
    'vit-large': ViTLarge_Embedder,
    'vit-huge': ViTHuge_Embedder
}