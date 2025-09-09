import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple
import scipy.io as sio


class HyperspectralImgDataset(Dataset):
    """
    Dataset for hyperspectral images with 31 channels
    Supports .mat files and other hyperspectral formats
    """
    
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        transform=None,
        spectral_channels: int = 31,
        load_format: str = "mat"  # "mat", "npy", "hdr"
    ):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform
        self.spectral_channels = spectral_channels
        self.load_format = load_format
        
        # Get file lists
        if load_format == "mat":
            self.source_files = sorted(list(self.source_dir.glob("*.mat")))
            self.target_files = sorted(list(self.target_dir.glob("*.mat")))
        elif load_format == "npy":
            self.source_files = sorted(list(self.source_dir.glob("*.npy")))
            self.target_files = sorted(list(self.target_dir.glob("*.npy")))
        else:
            raise ValueError(f"Unsupported format: {load_format}")
        
        assert len(self.source_files) == len(self.target_files), \
            f"Mismatch in number of source ({len(self.source_files)}) and target ({len(self.target_files)}) files"
    
    def __len__(self) -> int:
        return len(self.source_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source_path = self.source_files[idx]
        target_path = self.target_files[idx]
        
        # Load hyperspectral data
        source_data = self._load_hyperspectral(source_path)
        target_data = self._load_hyperspectral(target_path)
        
        # Convert to tensors
        source_tensor = torch.from_numpy(source_data).float()
        target_tensor = torch.from_numpy(target_data).float()
        
        # Ensure correct channel dimension (C, H, W)
        if source_tensor.ndim == 3 and source_tensor.shape[2] == self.spectral_channels:
            source_tensor = source_tensor.permute(2, 0, 1)  # H, W, C -> C, H, W
        if target_tensor.ndim == 3 and target_tensor.shape[2] == self.spectral_channels:
            target_tensor = target_tensor.permute(2, 0, 1)  # H, W, C -> C, H, W
        
        # Apply transforms if provided
        if self.transform:
            source_tensor = self.transform(source_tensor)
            target_tensor = self.transform(target_tensor)
        
        return source_tensor, target_tensor
    
    def _load_hyperspectral(self, file_path: Path) -> np.ndarray:
        """Load hyperspectral data from file"""
        if self.load_format == "mat":
            # Load .mat file
            mat_data = sio.loadmat(str(file_path))
            
            # Try common field names for hyperspectral data
            possible_keys = ['data', 'hyperspectral', 'hsi', 'img', 'image']
            data = None
            
            for key in possible_keys:
                if key in mat_data:
                    data = mat_data[key]
                    break
            
            if data is None:
                # If no common key found, use the first non-metadata key
                for key, value in mat_data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray):
                        data = value
                        break
            
            if data is None:
                raise ValueError(f"Could not find hyperspectral data in {file_path}")
                
        elif self.load_format == "npy":
            data = np.load(str(file_path))
        else:
            raise ValueError(f"Unsupported format: {self.load_format}")
        
        # Ensure data has correct shape and channels
        if data.ndim == 3:
            if data.shape[0] == self.spectral_channels:
                # Already in C, H, W format
                data = data.transpose(1, 2, 0)  # Convert to H, W, C for consistency
            elif data.shape[2] == self.spectral_channels:
                # Already in H, W, C format
                pass
            else:
                raise ValueError(f"Expected {self.spectral_channels} channels, got shape {data.shape}")
        else:
            raise ValueError(f"Expected 3D data, got shape {data.shape}")
        
        # Normalize to [0, 1] range if needed
        if data.max() > 1.0:
            data = data / data.max()
        
        return data.astype(np.float32)
