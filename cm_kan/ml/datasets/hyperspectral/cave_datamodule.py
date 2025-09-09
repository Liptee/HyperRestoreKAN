import lightning as L
from torch.utils.data import DataLoader
from .cave_dataset import CAVEDataset, CAVEPredictDataset
import torch


class CAVEDataModule(L.LightningDataModule):
    """
    Lightning DataModule for CAVE hyperspectral dataset
    Handles the specific format and structure of CAVE database
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
        spectral_channels: int = 31
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

    def setup(self, stage: str = None):
        """Setup datasets for different stages"""
        
        # Training dataset
        if stage == "fit" or stage is None:
            self.train_dataset = CAVEDataset(
                source_dir=self.train_a,
                target_dir=self.train_b,
                spectral_channels=self.spectral_channels,
                transform=self._get_train_transform()
            )
            
            self.val_dataset = CAVEDataset(
                source_dir=self.val_a,
                target_dir=self.val_b,
                spectral_channels=self.spectral_channels,
                transform=self._get_val_transform()
            )
        
        # Test dataset
        if stage == "test" or stage is None:
            self.test_dataset = CAVEDataset(
                source_dir=self.test_a,
                target_dir=self.test_b,
                spectral_channels=self.spectral_channels,
                transform=self._get_val_transform()
            )
        
        if stage == "predict" or stage is None:
            self.predict_dataset = CAVEPredictDataset(
                source_dir=self.test_a,  # Use test data for prediction
                spectral_channels=self.spectral_channels,
                transform=self._get_val_transform()
            )

    def train_dataloader(self):
        # Debug: print(f"Creating train dataloader: batch_size={self.batch_size}, num_workers={self.num_workers}")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False if self.num_workers == 0 else True,  # Disable pin_memory when num_workers=0
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False if self.num_workers == 0 else True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False if self.num_workers == 0 else True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False if self.num_workers == 0 else True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def _get_train_transform(self):
        """Get training transforms for CAVE hyperspectral data"""
        return HyperspectralTrainTransform()

    def _get_val_transform(self):
        """Get validation transforms for CAVE hyperspectral data"""
        return HyperspectralValTransform()


class HyperspectralTrainTransform:
    """Picklable training transform for hyperspectral data"""
    
    def __call__(self, x):
        # Spectral normalization - normalize each channel independently
        for c in range(x.shape[0]):
            channel = x[c]
            mean = channel.mean()
            std = channel.std()
            if std > 1e-8:
                x[c] = (channel - mean) / std
            else:
                x[c] = channel - mean
        
        # Clip to reasonable range
        x = torch.clamp(x, -3, 3)
        
        # Optional: Add some spectral augmentation
        # Slight spectral shift (simulates sensor variations)
        if x.shape[0] == 31:  # Only for hyperspectral data
            spectral_noise = torch.randn_like(x) * 0.02
            x = x + spectral_noise
        
        return x


class HyperspectralValTransform:
    """Picklable validation transform for hyperspectral data"""
    
    def __call__(self, x):
        # Same normalization as training but without augmentation
        for c in range(x.shape[0]):
            channel = x[c]
            mean = channel.mean()
            std = channel.std()
            if std > 1e-8:
                x[c] = (channel - mean) / std
            else:
                x[c] = channel - mean
        
        x = torch.clamp(x, -3, 3)
        return x
