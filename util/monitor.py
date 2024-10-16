# training monitor
from typing import Any, Dict
import torch

class TrainingMonitor:
    def __init__(self):
        self.num_updates = 0
        self.metrics = {}
        self.avg_metrics = None

    def update(self, logs: Dict):
        for key, value in logs.items():
            if key not in self.metrics:
                self.metrics[key] = 0
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key] += value
        self.num_updates += 1

    def add(self, key: str, value: Any = None):
        if value is None:
            assert key not in self.metrics, f"{key} already exists in the monitor!"
            self.metrics[key] = 0
        else:
            self.metrics[key] += value
    
    def get_metric(self,):
        return self.metrics

    def get_keys(self):
        # get what are monitored
        return list(self.metrics.keys())

    def get_avg(self):
        self.avg_metrics = {}
        if "training_steps" in self.metrics:
            running_steps = self.metrics["training_steps"]
        elif "steps" in self.metrics:
            running_steps = self.metrics["steps"]
        else:
            running_steps = self.num_updates
        for key, value in self.metrics.items():
            if key == "training_steps" or key == "steps":
                continue
            self.avg_metrics[key] = value / (1.0 * running_steps)
        return self.avg_metrics

    def reset(self):
        for key in self.metrics.keys():
            self.metrics[key] = 0
        self.num_updates = 0

    def export_logging(self):
        # Formatt as string, export to logger
        logging_str = str()
        # logging_str = f"Train Loss: {self.avg_metrics['running_loss_total']:.4f}, Train Obj Loss: {self.avg_metrics['running_loss_obj']:.4f}, Train Pr Loss: {self.avg_metrics['running_loss_pr']:.4f}"
        if 'running_loss_total' in self.avg_metrics:
            logging_str += f"Total Loss: {self.avg_metrics['running_loss_total']:.4f}, "
        if 'running_loss_obj' in self.avg_metrics:
            logging_str += f"Obj Loss: {self.avg_metrics['running_loss_obj']:.4f}, " 
        if 'running_loss_pr' in self.avg_metrics:
            logging_str += f"Pr Loss: {self.avg_metrics['running_loss_pr']:.4f}, "
        if "tcr" in self.avg_metrics:
            logging_str += f" -TCR: {self.avg_metrics['tcr']:.4f}"
        if "obj_sim_loss" in self.avg_metrics:
            logging_str += f" Object Sim Loss: {self.avg_metrics['obj_sim_loss']:.4f}"
        return logging_str

    def export_wandb(self):
        # Formatt as dict, export to wandb
        # For example, to log with WandB:
        # wandb.log(self.get_avg())
        return self.avg_metrics