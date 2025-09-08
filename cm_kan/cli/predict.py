import argparse
import yaml
from ..core import Logger
from ..core.selector import ModelSelector, PipelineSelector
from ..core.config import Config
from ..core.config.pipeline import PipelineType
from ..ml.datasets import ImgPredictDataModule, HyperspectralPredictDataModule
import lightning as L
import os
from lightning.pytorch.callbacks import (
    RichModelSummary,
    RichProgressBar,
)
from cm_kan.ml.callbacks import HyperspectralPredictionWriter
from lightning.pytorch.loggers import CSVLogger
from cm_kan import cli


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "predict",
        help="Process images with a trained model",
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
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Path to checkpoint file in the experiment folder",
        default="logs/checkpoints/last.ckpt",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the input image folder",
        default="data/samples/input",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        help=(
            "Path to the reference image folder (only for pair-based pipeline)"
        ),
        default="data/samples/reference",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output folder, will be created if not exists",
        default="data/samples/output",
        required=False,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        help="Batch size for prediction",
        default=1,
        required=False,
    )

    parser.set_defaults(func=predict)


def predict(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.input):
        raise ValueError(
            f"Incorrect input path '{args.input}'. It should be a directory."
        )

    Logger.info(f"Loading config from '{args.config}'")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = Config(**config)

    # No need to check reference path for hyperspectral supervised pipeline

    # Always use inference mode for hyperspectral supervised pipeline
    inference_mode = True
    Logger.info("Config:")
    config.print()

    # Use HyperspectralPredictDataModule for hyperspectral data (.mat files)
    dm = HyperspectralPredictDataModule(
        input_path=args.input,
        reference_path=args.reference,
        pipeline_type=config.pipeline.type,
        batch_size=args.batch_size,
        spectral_channels=31,  # CAVE dataset has 31 spectral channels
    )
    model = ModelSelector.select(config)
    pipeline = PipelineSelector.select(config, model)

    logger = CSVLogger(
        save_dir=os.path.join(config.save_dir, config.experiment),
        name="logs",
        version="",
    )

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=os.path.join(config.save_dir, config.experiment),
        max_epochs=config.pipeline.params.epochs,
        accelerator=config.accelerator,
        callbacks=[
            RichModelSummary(),
            RichProgressBar(),
            HyperspectralPredictionWriter(
                output_dir=os.path.join(args.output),
                write_interval="batch",
            ),
        ],
        inference_mode=inference_mode,
    )

    ckpt_path = args.weights
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint file '{ckpt_path}' does not exist.")
    trainer.predict(
        model=pipeline,
        datamodule=dm,
        ckpt_path=ckpt_path,
        return_predictions=False,
    )
