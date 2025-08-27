from ..config.data import DataType
from ..config import Config
from typing import Union
from cm_kan.ml.datasets import (
    HyperspectralImgDataModule,
    CAVEDataModule,
)


class DataSelector:
    def select(config: Config) -> Union[
        HyperspectralImgDataModule,
        CAVEDataModule
    ]:
        match config.data.type:
            case DataType.hyperspectral:
                return HyperspectralImgDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=config.data.val.source,
                    val_b=config.data.val.target,
                    test_a=config.data.test.source,
                    test_b=config.data.test.target,
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                    spectral_channels=31,
                    load_format="mat"
                )
            case DataType.cave:
                # Safely extract hardware config
                hardware_config = getattr(config, 'hardware', None)
                if hardware_config and hasattr(hardware_config, 'num_workers'):
                    num_workers = hardware_config.num_workers
                else:
                    # Use a safe default for CAVE dataset
                    num_workers = 0  # Set to 0 for debugging and memory efficiency
                
                # Debug: print(f"DataSelector: Creating CAVEDataModule with num_workers={num_workers}")
                return CAVEDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=config.data.val.source,
                    val_b=config.data.val.target,
                    test_a=config.data.test.source,
                    test_b=config.data.test.target,
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                    num_workers=num_workers,
                    spectral_channels=31
                )
            case _:
                raise ValueError(f"Unsupported data type {config.data.type}")
