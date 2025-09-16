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
from cm_kan.ml.callbacks import GenerateCallback, TelegramNotificationCallback
from lightning.pytorch.loggers import CSVLogger
from cm_kan import cli
import torch
from datetime import timedelta


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

    # Set NCCL environment variables for better stability
    os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 minutes timeout
    os.environ.setdefault('NCCL_HEARTBEAT_TIMEOUT_SEC', '300')  # 5 minutes heartbeat
    os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')  # Enable blocking wait
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # Enable async error handling
    
    # For Docker environments with limited GPU visibility
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        Logger.info(f"Available GPUs: {available_gpus}")
        
        # If running in Docker with limited GPU access, set CUDA_VISIBLE_DEVICES
        if 'CUDA_VISIBLE_DEVICES' not in os.environ and available_gpus > 4:
            # Assume we want to use the last 4 GPUs (4,5,6,7) as specified in Docker run
            os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
            Logger.info("Set CUDA_VISIBLE_DEVICES=4,5,6,7 for Docker environment")

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
            strategy = DDPStrategy(
                find_unused_parameters=True,
                timeout=timedelta(minutes=30),
                process_group_backend="nccl"
            )
        elif config.strategy == "ddp":
            strategy = DDPStrategy(
                find_unused_parameters=False,
                timeout=timedelta(minutes=30),
                process_group_backend="nccl"
            )
        else:
            strategy = config.strategy
    else:
        # Auto-detect if we need DDP with unused parameters
        if (isinstance(devices, int) and devices > 1) or os.environ.get('WORLD_SIZE', '1') != '1':
            Logger.info("Multi-GPU/multi-node environment detected, using DDP with find_unused_parameters=True")
            strategy = DDPStrategy(
                find_unused_parameters=True,
                timeout=timedelta(minutes=30),
                process_group_backend="nccl"
            )
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
                filename="{epoch}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_last=True,
                save_top_k=-1,  # Save all checkpoints (every epoch)
                every_n_epochs=1,  # Save every epoch
            ),
            RichModelSummary(),
            RichProgressBar(),
            LearningRateMonitor(
                logging_interval="epoch",
            ),
            GenerateCallback(
                every_n_epochs=1,
            ),
            TelegramNotificationCallback(
                script_path="/app/push.sh",
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
