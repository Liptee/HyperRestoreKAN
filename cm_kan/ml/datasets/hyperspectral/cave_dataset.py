import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import scipy.io as sio
from PIL import Image


class CAVEDataset(Dataset):
    """
    Dataset for CAVE hyperspectral database
    Handles the specific format of CAVE dataset with 31 spectral bands
    """
    
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        transform=None,
        spectral_channels: int = 31
    ):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform
        self.spectral_channels = spectral_channels
        
        # Get file lists - CAVE dataset typically uses .mat files
        self.source_files = sorted(list(self.source_dir.glob("*.mat")))
        self.target_files = sorted(list(self.target_dir.glob("*.mat")))
        
        # If no .mat files, try .png files (for RGB versions)
        if len(self.source_files) == 0:
            self.source_files = sorted(list(self.source_dir.glob("*.png")))
        if len(self.target_files) == 0:
            self.target_files = sorted(list(self.target_dir.glob("*.mat")))
        
        # Debug information (only if no files found)
        if len(self.source_files) == 0 or len(self.target_files) == 0:
            print(f"CAVEDataset Debug:")
            print(f"  Source dir: {self.source_dir}")
            print(f"  Target dir: {self.target_dir}")
            print(f"  Source files found: {len(self.source_files)}")
            print(f"  Target files found: {len(self.target_files)}")
            if len(self.source_files) > 0:
                print(f"  First source file: {self.source_files[0]}")
            if len(self.target_files) > 0:
                print(f"  First target file: {self.target_files[0]}")
        
        assert len(self.source_files) == len(self.target_files), \
            f"Mismatch in number of source ({len(self.source_files)}) and target ({len(self.target_files)}) files"
        
        if len(self.source_files) == 0:
            raise ValueError(f"No files found in source directory: {self.source_dir}")
        if len(self.target_files) == 0:
            raise ValueError(f"No files found in target directory: {self.target_dir}")
    
    def __len__(self) -> int:
        return len(self.source_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source_path = self.source_files[idx]
        target_path = self.target_files[idx]
        
        # Load data based on file extension
        if source_path.suffix.lower() == '.mat':
            source_data = self._load_mat_file(source_path)
        elif source_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            source_data = self._load_rgb_file(source_path)
        else:
            raise ValueError(f"Unsupported source file format: {source_path.suffix}")
        
        if target_path.suffix.lower() == '.mat':
            target_data = self._load_mat_file(target_path)
        else:
            raise ValueError(f"Unsupported target file format: {target_path.suffix}")
        
        # Convert to tensors
        source_tensor = torch.from_numpy(source_data).float()
        target_tensor = torch.from_numpy(target_data).float()
        
        # Ensure correct channel dimension (C, H, W)
        if source_tensor.ndim == 3:
            if source_tensor.shape[2] in [3, 31]:  # H, W, C format
                source_tensor = source_tensor.permute(2, 0, 1)  # H, W, C -> C, H, W
        
        if target_tensor.ndim == 3:
            if target_tensor.shape[2] == 31:  # H, W, C format
                target_tensor = target_tensor.permute(2, 0, 1)  # H, W, C -> C, H, W
        
        # Apply transforms if provided
        if self.transform:
            source_tensor = self.transform(source_tensor)
            target_tensor = self.transform(target_tensor)
        
        return source_tensor, target_tensor
    
    def _load_mat_file(self, file_path: Path) -> np.ndarray:
        """Load hyperspectral data from .mat file"""
        mat_data = sio.loadmat(str(file_path))
        
        # CAVE dataset typically stores data in specific fields
        possible_keys = ['hyperspectral', 'hsi', 'data', 'img', 'image', 'cube']
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
            raise ValueError(f"Could not find hyperspectral data in {file_path}")
        
        # Ensure correct shape
        if data.ndim == 3:
            # Check if we need to transpose
            if data.shape[0] == 31:
                # C, H, W -> H, W, C
                data = data.transpose(1, 2, 0)
            elif data.shape[2] != 31 and data.shape[0] != 31:
                # Try to find the correct axis with 31 channels
                if data.shape[1] == 31:
                    data = data.transpose(0, 2, 1)  # H, C, W -> H, W, C
        
        # Normalize to [0, 1] range
        if data.max() > 1.0:
            data = data / data.max()
        
        return data.astype(np.float32)
    
    def _load_rgb_file(self, file_path: Path) -> np.ndarray:
        """Load RGB image and convert to 3-channel tensor"""
        img = Image.open(file_path).convert('RGB')
        data = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        return data  # H, W, C format
