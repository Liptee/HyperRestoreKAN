from pydantic import BaseModel
from enum import Enum
from typing import List, Union


class ModelType(str, Enum):
    hyperspectral_cm_kan = "hyperspectral_cm_kan"
    light_hyperspectral_cm_kan = "light_hyperspectral_cm_kan"
    hyperspectral_cycle_cm_kan = "hyperspectral_cycle_cm_kan"
    multiscale_hyperspectral_cm_kan = "multiscale_hyperspectral_cm_kan"
    minimal_hyperspectral_cm_kan = "minimal_hyperspectral_cm_kan"


class HyperspectralCmKanModelParams(BaseModel):
    in_dims: List[int] = [31]
    out_dims: List[int] = [31]
    grid_size: int = 5
    spline_order: int = 3
    residual_std: float = 0.1
    grid_range: List[float] = [-1.0, 1.0]
    use_spectral_processor: bool = True
    use_residual_blocks: bool = True
    num_residual_blocks: int = 2
    use_gradient_checkpointing: bool = True


class MultiScaleHyperspectralCmKanModelParams(BaseModel):
    in_dims: List[int] = [31]
    out_dims: List[int] = [31]
    scales: List[float] = [1.0, 0.75, 0.5]
    grid_size: int = 5
    spline_order: int = 3
    residual_std: float = 0.1
    grid_range: List[float] = [-1.0, 1.0]


class MinimalHyperspectralCmKanModelParams(BaseModel):
    in_dims: List[int] = [31]
    out_dims: List[int] = [31]
    grid_size: int = 2
    spline_order: int = 1
    residual_std: float = 0.1
    grid_range: List[float] = [-1.0, 1.0]
    chunk_size: int = 512


class Model(BaseModel):
    type: ModelType = ModelType.hyperspectral_cm_kan
    params: Union[
        HyperspectralCmKanModelParams,
        MultiScaleHyperspectralCmKanModelParams,
        MinimalHyperspectralCmKanModelParams
    ] = HyperspectralCmKanModelParams()
