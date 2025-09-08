from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
import torch
import scipy.io as sio
import os
import numpy as np


class SaveMatPredictionsCallback(Callback):
    def __init__(self, save_dir=None) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.predictions = []
        self.targets = []
        self.inputs = []
        
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize save directory at test start"""
        if self.save_dir is None:
            self.save_dir = os.path.join(trainer.log_dir, "mat_predictions")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Clear any previous data
        self.predictions = []
        self.targets = []
        self.inputs = []
        
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, 
                         outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Collect predictions, targets, and inputs from each batch"""
        inputs, targets = batch
        
        # Get predictions for this batch
        with torch.no_grad():
            pl_module.eval()
            predictions = pl_module(inputs)
            
        # Move to CPU and convert to numpy
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        inputs_np = inputs.detach().cpu().numpy()
        
        # Store batch data
        self.predictions.append(predictions_np)
        self.targets.append(targets_np)
        self.inputs.append(inputs_np)
        
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save all collected predictions to mat files"""
        if not self.predictions:
            return
            
        # Concatenate all batches
        all_predictions = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        all_inputs = np.concatenate(self.inputs, axis=0)
        
        # Save predictions as mat file
        predictions_path = os.path.join(self.save_dir, "predictions.mat")
        sio.savemat(predictions_path, {
            'predictions': all_predictions,
            'targets': all_targets,
            'inputs': all_inputs,
            'shape_info': {
                'batch_size': all_predictions.shape[0],
                'channels': all_predictions.shape[1],
                'height': all_predictions.shape[2],
                'width': all_predictions.shape[3]
            }
        })
        
        print(f"Saved predictions to {predictions_path}")
        print(f"Shape: {all_predictions.shape}")
        
        # Also save individual samples if there are multiple samples
        if all_predictions.shape[0] > 1:
            samples_dir = os.path.join(self.save_dir, "individual_samples")
            os.makedirs(samples_dir, exist_ok=True)
            
            for i in range(min(all_predictions.shape[0], 10)):  # Save first 10 samples
                sample_path = os.path.join(samples_dir, f"sample_{i:03d}.mat")
                sio.savemat(sample_path, {
                    'prediction': all_predictions[i],
                    'target': all_targets[i],
                    'input': all_inputs[i]
                })
                
        print(f"Saved {len(self.predictions)} batches of predictions to mat files")

