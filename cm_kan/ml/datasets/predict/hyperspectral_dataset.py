import torch
import os
from torch.utils.data import Dataset
import numpy as np
from typing import List
import scipy.io as sio


class HyperspectralPredictDataset(Dataset):
    """Dataset for hyperspectral prediction from .mat files"""
    
    def __init__(self, paths: List[str], spectral_channels: int = 31) -> None:
        self.paths = paths
        self.spectral_channels = spectral_channels
        
        # Debug: print the paths to see what we're working with
        print(f"HyperspectralPredictDataset initialized with {len(paths)} files:")
        for i, path in enumerate(paths[:5]):  # Show first 5 files
            print(f"  {i}: {path}")
        if len(paths) > 5:
            print(f"  ... and {len(paths) - 5} more files")

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        path = self.paths[idx]
        filename = os.path.basename(path)
        
        # Load hyperspectral data from .mat file
        x = self._load_hyperspectral_mat(path)
        
        # Convert to tensor and ensure correct format (C, H, W)
        x_tensor = torch.from_numpy(x)
        if x_tensor.dim() == 3:
            # If it's H, W, C, convert to C, H, W
            if x_tensor.shape[2] == self.spectral_channels:
                x_tensor = x_tensor.permute(2, 0, 1)  # H, W, C -> C, H, W
        
        return filename, x_tensor

    def __len__(self) -> int:
        return len(self.paths)
    
    def _load_hyperspectral_mat(self, file_path: str) -> np.ndarray:
        """Load hyperspectral data from .mat file"""
        try:
            mat_data = sio.loadmat(file_path)
        except Exception as e:
            raise ValueError(f"Could not load .mat file {file_path}: {e}")
        
        # Try common field names for hyperspectral data
        possible_keys = ['hyperspectral', 'hsi', 'data', 'img', 'image', 'cube', 'rad']
        data = None
        
        for key in possible_keys:
            if key in mat_data:
                data = mat_data[key]
                break
        
        if data is None:
            # If no common key found, use the first non-metadata key
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 3:
                    data = value
                    break
        
        if data is None:
            raise ValueError(f"Could not find hyperspectral data in {file_path}. Available keys: {list(mat_data.keys())}")
        
        # Ensure correct shape and channels
        if data.ndim == 3:
            # Check if we need to transpose to get the right channel dimension
            if data.shape[0] == self.spectral_channels:
                # Already in C, H, W format, convert to H, W, C for consistency
                data = data.transpose(1, 2, 0)
            elif data.shape[2] == self.spectral_channels:
                # Already in H, W, C format
                pass
            elif data.shape[1] == self.spectral_channels:
                # H, C, W format, convert to H, W, C
                data = data.transpose(0, 2, 1)
            else:
                # Try to infer the correct format
                print(f"Warning: Unexpected shape {data.shape} for {file_path}. Expected {self.spectral_channels} channels.")
                # Assume the last dimension is channels if it's reasonable
                if data.shape[2] <= 64:  # Reasonable number of channels
                    pass  # Keep as H, W, C
                elif data.shape[0] <= 64:
                    data = data.transpose(1, 2, 0)  # C, H, W -> H, W, C
                else:
                    raise ValueError(f"Cannot determine channel dimension for shape {data.shape}")
        else:
            raise ValueError(f"Expected 3D data, got shape {data.shape}")
        
        # Normalize to [0, 1] range if needed
        if data.max() > 1.0:
            data = data / data.max()
        
        return data.astype(np.float32)
