from typing import Union, Optional
from pydantic import BaseModel

from rich.syntax import Syntax
from rich import print

from .data import Data
from .model import Model
from .pipeline import Pipeline

import yaml


class HardwareConfig(BaseModel):
    num_workers: int = 0
    pin_memory: bool = False
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1


class Config(BaseModel):
    experiment: str = "volga2k_supervised"
    save_dir: str = "experiments"
    resume: bool = False
    model: Model
    data: Data
    pipeline: Pipeline
    accelerator: Union[str, int] = "gpu"
    devices: Union[int, str] = "auto"
    strategy: str = "auto"  # Can be "auto", "ddp", "ddp_find_unused_parameters_true", etc.
    hardware: Optional[HardwareConfig] = None

    def print(self) -> None:
        str = yaml.dump(self.model_dump())
        syntax = Syntax(str, "yaml")
        print(syntax)
