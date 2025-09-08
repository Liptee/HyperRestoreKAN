from lightning.pytorch.callbacks import BasePredictionWriter
from lightning import LightningModule, Trainer
from typing import List, Tuple
import torch
import os
import scipy.io as sio
import numpy as np


class HyperspectralPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str = "batch") -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.write_interval = write_interval

        os.makedirs(output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: torch.Tensor,
        batch_indices: List[int],
        batch: Tuple[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        paths = batch[0]
        for i, path in enumerate(paths):
            # Get the prediction for this sample
            pred_tensor = prediction[i].cpu().numpy()  # Shape: (C, H, W)
            
            # Transpose from (C, H, W) to (H, W, C) for MATLAB format
            if pred_tensor.ndim == 3:
                pred_data = pred_tensor.transpose(1, 2, 0)  # (H, W, C)
            else:
                pred_data = pred_tensor
            
            # Create output filename (change extension to .mat)
            base_name = os.path.splitext(path)[0]
            output_path = os.path.join(self.output_dir, f"{base_name}.mat")
            
            # Save as .mat file
            sio.savemat(output_path, {
                'hyperspectral': pred_data,
                'shape': pred_data.shape,
                'channels': pred_data.shape[-1] if pred_data.ndim == 3 else 1
            })
            
            print(f"Saved prediction to: {output_path}")


