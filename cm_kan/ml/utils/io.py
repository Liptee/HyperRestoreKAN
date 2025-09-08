import imageio.v3 as imageio
import numpy as np
import scipy.io as sio
import os


def read_bayer_image(path: str):
    raw = imageio.imread(path)
    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    return np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))


def read_rgb_image(path: str) -> np.ndarray:
    return imageio.imread(path)


def read_numpy_feature(path: str) -> np.ndarray:
    return np.load(path)


def read_hyperspectral_mat(path: str) -> np.ndarray:
    """Load hyperspectral data from .mat file"""
    mat_data = sio.loadmat(str(path))
    
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
        raise ValueError(f"Could not find hyperspectral data in {path}")
    
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


def read_image_or_mat(path: str) -> np.ndarray:
    """Read image or mat file based on extension"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        return read_hyperspectral_mat(path)
    else:
        return imageio.imread(path)
