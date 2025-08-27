from pydantic import BaseModel
from enum import Enum


class DataType(str, Enum):
    hyperspectral = "hyperspectral"
    cave = "cave"  # CAVE hyperspectral dataset


class DataPathes(BaseModel):
    source: str
    target: str


class Data(BaseModel):
    type: DataType = DataType.hyperspectral
    train: DataPathes
    val: DataPathes
    test: DataPathes
