import lightning as L
from torch.utils.data import DataLoader
from .img_dataset import HyperspectralImgDataset
import torch


class HyperspectralImgDataModule(L.LightningDataModule):
    """
    Lightning DataModule for hyperspectral image datasets
    """
    
    def __init__(
        self,
        train_a: str,
        train_b: str,
        val_a: str,
        val_b: str,
        test_a: str,
        test_b: str,
        batch_size: int = 8,
        val_batch_size: int = 4,
        test_batch_size: int = 4,
        num_workers: int = 4,
        spectral_channels: int = 31,
        load_format: str = "mat"
    ):
        super().__init__()
        self.train_a = train_a
        self.train_b = train_b
        self.val_a = val_a
        self.val_b = val_b
        self.test_a = test_a
        self.test_b = test_b
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.spectral_channels = spectral_channels
        self.load_format = load_format

    def setup(self, stage: str = None):
        """Setup datasets for different stages"""
        
        # Training dataset
        if stage == "fit" or stage is None:
            self.train_dataset = HyperspectralImgDataset(
                source_dir=self.train_a,
                target_dir=self.train_b,
                spectral_channels=self.spectral_channels,
                load_format=self.load_format,
                transform=self._get_train_transform()
            )
            
            self.val_dataset = HyperspectralImgDataset(
                source_dir=self.val_a,
                target_dir=self.val_b,
                spectral_channels=self.spectral_channels,
                load_format=self.load_format,
                transform=self._get_val_transform()
            )
        
        # Test dataset
        if stage == "test" or stage is None:
            self.test_dataset = HyperspectralImgDataset(
                source_dir=self.test_a,
                target_dir=self.test_b,
                spectral_channels=self.spectral_channels,
                load_format=self.load_format,
                transform=self._get_val_transform()
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def _get_train_transform(self):
        """Get training transforms for hyperspectral data"""
        return BasicHyperspectralTransform()

    def _get_val_transform(self):
        """Get validation transforms for hyperspectral data"""
        return BasicHyperspectralTransform()


class BasicHyperspectralTransform:
    """Picklable basic transform for hyperspectral data"""
    
    def __call__(self, x):
        # Basic normalization
        x = (x - x.mean()) / (x.std() + 1e-8)
        # Clip to reasonable range
        x = torch.clamp(x, -3, 3)
        return x
