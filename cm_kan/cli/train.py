import argparse
import yaml
from ..core import Logger
from ..core.selector import ModelSelector, DataSelector, PipelineSelector
from ..core.config import Config
import lightning as L
import os
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)
from lightning.pytorch.strategies import DDPStrategy
from cm_kan.ml.callbacks import GenerateCallback
from lightning.pytorch.loggers import CSVLogger
from cm_kan import cli


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "train",
        help="Train color transfer model",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to config file",
        default="config.yaml",
        required=False,
    )

    parser.set_defaults(func=train)


def train(args: argparse.Namespace) -> None:
    Logger.info(f"Loading config from '{args.config}'")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = Config(**config)
    Logger.info("Config:")
    config.print()

    dm = DataSelector.select(config)
    model = ModelSelector.select(config)
    pipeline = PipelineSelector.select(config, model)

    logger = CSVLogger(
        save_dir=os.path.join(config.save_dir, config.experiment),
        name="logs",
        version="",
    )

    # Configure strategy for DDP with unused parameters handling
    devices = config.devices if hasattr(config, 'devices') else "auto"
    
    # Handle strategy configuration - always set a valid strategy
    if hasattr(config, 'strategy') and config.strategy and config.strategy != "auto":
        if config.strategy == "ddp_find_unused_parameters_true":
            strategy = DDPStrategy(find_unused_parameters=True)
        elif config.strategy == "ddp":
            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            strategy = config.strategy
    else:
        # Auto-detect if we need DDP with unused parameters
        if (isinstance(devices, int) and devices > 1) or os.environ.get('WORLD_SIZE', '1') != '1':
            Logger.info("Multi-GPU/multi-node environment detected, using DDP with find_unused_parameters=True")
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            # Default to auto for single device
            strategy = "auto"

    Logger.info(f"Training configuration: devices={devices}, strategy={strategy}")

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=os.path.join(config.save_dir, config.experiment),
        max_epochs=config.pipeline.params.epochs,
        accelerator=config.accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{val_de:.2f}",
                monitor="val_de",
                save_last=True,
            ),
            RichModelSummary(),
            RichProgressBar(),
            LearningRateMonitor(
                logging_interval="epoch",
            ),
            GenerateCallback(
                every_n_epochs=1,
            ),
        ],
    )

    ckpt_path = os.path.join(
        config.save_dir, config.experiment, "logs/checkpoints/last.ckpt"
    )

    trainer.fit(
        model=pipeline,
        datamodule=dm,
        ckpt_path=(
            ckpt_path if config.resume and os.path.exists(ckpt_path) else None
        ),
    )
