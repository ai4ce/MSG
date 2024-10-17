# checkpointing
# save and load
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def save_checkpoint(model: nn.Module, optimizer: torch.optim, checkpoint_path: str, config: Optional[Dict[str, Any]] = None) -> None:
    checkpoint = {
        'config': config, 
        'state_dict': {},
        'optimizer': optimizer.state_dict(),
    }
    for module_name, module in model.named_children():
        if hasattr(module, 'parameters') and sum(1 for _ in module.parameters()) > 0:
            # Check if any parameter in the module requires gradient update
            if any(p.requires_grad for p in module.parameters()):
                checkpoint['state_dict'][module_name] = module.state_dict()
            else:
                # Mark the module as not requiring updates
                checkpoint['state_dict'][module_name] = 'frozen'
        else:
            # Handle special case for the detector module
            if module_name == 'detector' and module is not None:
                checkpoint['state_dict'][module_name] = 'frozen'
            else:
                checkpoint['state_dict'][module_name] = 'No param'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model: nn.Module, checkpoint_path: str, optimizer: torch.optim = None, logger: logging.Logger = None) -> None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    for module_name, module in model.named_children():
        if module_name in checkpoint['state_dict']:
            state_info = checkpoint['state_dict'][module_name]
            if isinstance(state_info, dict):  # The module needs to be updated
                module.load_state_dict(state_info, strict=False)
                logger.info(f"load {module_name}")
            else:
                print(module_name, state_info)
                logger.info(f"{module_name} == {state_info}")

