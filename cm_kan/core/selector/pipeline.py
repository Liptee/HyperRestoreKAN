from ..config.pipeline import PipelineType
from ..config import Config
from typing import Union
from cm_kan.ml.pipelines import SupervisedPipeline
from cm_kan.ml.models import (
    HyperspectralCmKAN,
    LightHyperspectralCmKAN,
    HyperspectralCycleCmKAN,
    MultiScaleHyperspectralCmKAN,
)
from cm_kan.ml.models.minimal_hyperspectral import MinimalHyperspectralCmKAN


class PipelineSelector:
    def select(
        config: Config, model: Union[
            HyperspectralCmKAN,
            LightHyperspectralCmKAN,
            HyperspectralCycleCmKAN,
            MultiScaleHyperspectralCmKAN,
            MinimalHyperspectralCmKAN
        ]
    ) -> SupervisedPipeline:
        match config.pipeline.type:
            case PipelineType.supervised:
                return SupervisedPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay,
                )
            case _:
                raise ValueError(f"Unsupported pipeline type {config.pipeline.type}")
