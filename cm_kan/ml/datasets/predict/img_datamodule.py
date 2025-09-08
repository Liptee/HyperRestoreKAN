import os
import torch
import lightning as L
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
)
from torch.utils.data import DataLoader
from typing import Tuple
from .img_dataset import ImagePredictDataset, ImagePairedPredictDataset
from cm_kan.core.config.pipeline import PipelineType


class ImgPredictDataModule(L.LightningDataModule):
    def __init__(
        self,
        input_path: str,
        reference_path: str = None,
        pipeline_type: PipelineType = PipelineType.supervised,
        batch_size: int = 4,
        img_exts: Tuple[str] = (".png", ".jpg", ".mat"),
        num_workers: int = min(12, os.cpu_count() - 1),
    ) -> None:
        super().__init__()
        self.predict_dataset = None

        input_paths = [
            os.path.join(input_path, fname)
            for fname in os.listdir(input_path)
            if fname.endswith(img_exts)
        ]
        self.input_paths = sorted(input_paths)

        # For supervised pipeline, we don't need reference paths
        self.reference_paths = None

        self.image_transform = Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
            ]
        )
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pipeline_type = pipeline_type

    def setup(self, stage: str) -> None:
        if stage == "predict" or stage is None:
            # For supervised pipeline, use single input dataset
            self.dataset = ImagePredictDataset(
                self.input_paths,
                self.image_transform,
            )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
