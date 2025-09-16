from ..config.model import ModelType
from ..config import Config
from typing import Union
from cm_kan.ml.models import (
    HyperspectralCmKAN,
    LightHyperspectralCmKAN,
    HyperspectralCycleCmKAN,
    MultiScaleHyperspectralCmKAN,
)
from cm_kan.ml.models.minimal_hyperspectral import MinimalHyperspectralCmKAN


class ModelSelector:
    def select(config: Config) -> Union[
        HyperspectralCmKAN,
        LightHyperspectralCmKAN,
        HyperspectralCycleCmKAN,
        MultiScaleHyperspectralCmKAN,
        MinimalHyperspectralCmKAN
    ]:
        match config.model.type:
            case ModelType.hyperspectral_cm_kan:
                return HyperspectralCmKAN(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range,
                    use_spectral_processor=getattr(config.model.params, 'use_spectral_processor', True),
                    use_residual_blocks=getattr(config.model.params, 'use_residual_blocks', True),
                    num_residual_blocks=getattr(config.model.params, 'num_residual_blocks', 2),
                    use_gradient_checkpointing=getattr(config.model.params, 'use_gradient_checkpointing', True),
                    # New spectral mode parameters
                    spectral_mode=getattr(config.model.params, 'spectral_mode', "global"),
                    window_size=getattr(config.model.params, 'window_size', 3),
                    padding_mode=getattr(config.model.params, 'padding_mode', "reflect"),
                    wavelengths=getattr(config.model.params, 'wavelengths', None),
                    shared_kan_params=getattr(config.model.params, 'shared_kan_params', True),
                )
            case ModelType.light_hyperspectral_cm_kan:
                return LightHyperspectralCmKAN(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range,
                )
            case ModelType.hyperspectral_cycle_cm_kan:
                return HyperspectralCycleCmKAN(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range,
                )
            case ModelType.multiscale_hyperspectral_cm_kan:
                return MultiScaleHyperspectralCmKAN(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    scales=getattr(config.model.params, 'scales', [1.0, 0.75, 0.5]),
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range,
                )
            case ModelType.minimal_hyperspectral_cm_kan:
                return MinimalHyperspectralCmKAN(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range,
                    chunk_size=getattr(config.model.params, 'chunk_size', 512),
                )
            case _:
                raise ValueError(f"Unsupported model type {config.model.type}")
