import os
import torch
import lightning as L
from torch.utils.data import DataLoader
from typing import Tuple
from .hyperspectral_dataset import HyperspectralPredictDataset
from cm_kan.core.config.pipeline import PipelineType


class HyperspectralPredictDataModule(L.LightningDataModule):
    """Data module for hyperspectral prediction from .mat files"""
    
    def __init__(
        self,
        input_path: str,
        reference_path: str = None,  # Not used for prediction
        pipeline_type: PipelineType = PipelineType.supervised,
        batch_size: int = 4,
        spectral_channels: int = 31,
        img_exts: Tuple[str] = (".mat",),
        num_workers: int = min(12, os.cpu_count() - 1),
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.batch_size = batch_size
        self.spectral_channels = spectral_channels
        self.num_workers = num_workers
        self.pipeline_type = pipeline_type
        
        # Find all .mat files in the input directory
        if not os.path.isdir(input_path):
            raise ValueError(f"Input path '{input_path}' is not a directory")
        
        input_paths = [
            os.path.join(input_path, fname)
            for fname in os.listdir(input_path)
            if fname.endswith(img_exts)
        ]
        self.input_paths = sorted(input_paths)
        
        print(f"HyperspectralPredictDataModule found {len(self.input_paths)} files in {input_path}")
        if len(self.input_paths) == 0:
            print(f"Warning: No files with extensions {img_exts} found in {input_path}")
            print(f"Available files: {os.listdir(input_path) if os.path.exists(input_path) else 'Directory does not exist'}")

    def setup(self, stage: str) -> None:
        if stage == "predict" or stage is None:
            self.dataset = HyperspectralPredictDataset(
                paths=self.input_paths,
                spectral_channels=self.spectral_channels
            )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
