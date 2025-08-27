import torch
from torch import nn
import lightning as L
from torch import optim
from typing import Union as ModelUnion
from ..models import (
    HyperspectralCmKAN,
    LightHyperspectralCmKAN,
    HyperspectralCycleCmKAN,
    MultiScaleHyperspectralCmKAN,
)
from ..models.minimal_hyperspectral import MinimalHyperspectralCmKAN
from cm_kan.core import Logger
from ..metrics import (
    PSNR,
    SSIM,
    DeltaE,
)


class SupervisedPipeline(L.LightningModule):
    def __init__(
        self,
        model: ModelUnion[
            HyperspectralCmKAN,
            LightHyperspectralCmKAN,
            HyperspectralCycleCmKAN,
            MultiScaleHyperspectralCmKAN,
            MinimalHyperspectralCmKAN
        ],
        optimiser: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0,
    ) -> None:
        super(SupervisedPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.mae_loss = nn.L1Loss(reduction="mean")
        self.ssim_loss = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        
        # Flag to track if we're dealing with hyperspectral data
        self.is_hyperspectral = False

        self.save_hyperparameters(ignore=["model"])

    def setup(self, stage: str) -> None:
        """
        Initialize model weights before training
        """
        if stage == "fit" or stage is None:
            for m in self.model.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        Logger.info(
            "Initialized model weights with [bold green]Supervised[/bold green] pipeline."
        )

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            optimizer = optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "adamw":
            optimizer = optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"unsupported optimizer_type: {self.optimizer_type}")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=1, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        loss = mae_loss + (1 - ssim_loss) * 0.15

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        
        # Check if this is hyperspectral data (more than 3 channels)
        if inputs.shape[1] > 3:
            self.is_hyperspectral = True
        
        mae_loss = self.mae_loss(predictions, targets)
        psnr_metric = self.psnr_metric(predictions, targets)
        ssim_metric = self.ssim_metric(predictions, targets)
        
        # Only compute Delta E for RGB data (3 channels)
        if not self.is_hyperspectral and inputs.shape[1] == 3:
            de_metric = self.de_metric(predictions, targets)
            self.log("val_de", de_metric, prog_bar=True, logger=True)
        else:
            # For hyperspectral data, compute spectral angle mapper instead
            sam_metric = self._compute_spectral_angle_mapper(predictions, targets)
            self.log("val_sam", sam_metric, prog_bar=True, logger=True)

        self.log("val_psnr", psnr_metric, prog_bar=True, logger=True)
        self.log("val_ssim", ssim_metric, prog_bar=True, logger=True)
        self.log("val_loss", mae_loss, prog_bar=True, logger=True)
        return {"loss": mae_loss}

    def _compute_spectral_angle_mapper(self, predictions, targets):
        """
        Compute Spectral Angle Mapper (SAM) for hyperspectral data
        """
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        pred_flat = predictions.view(predictions.shape[0], predictions.shape[1], -1)
        target_flat = targets.view(targets.shape[0], targets.shape[1], -1)
        
        # Compute dot product and norms
        dot_product = torch.sum(pred_flat * target_flat, dim=1)  # (B, H*W)
        pred_norm = torch.norm(pred_flat, dim=1)  # (B, H*W)
        target_norm = torch.norm(target_flat, dim=1)  # (B, H*W)
        
        # Avoid division by zero
        norms_product = pred_norm * target_norm
        norms_product = torch.clamp(norms_product, min=1e-8)
        
        # Compute cosine similarity and then angle
        cos_angle = torch.clamp(dot_product / norms_product, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        
        # Return mean angle in degrees
        return torch.mean(angle) * 180.0 / torch.pi

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)
        psnr_metric = self.psnr_metric(predictions, targets)  # Fixed typo: panr -> psnr
        ssim_metric = self.ssim_metric(predictions, targets)
        
        # Only compute Delta E for RGB data (3 channels)
        if not self.is_hyperspectral and inputs.shape[1] == 3:
            de_metric = self.de_metric(predictions, targets)
            self.log("test_de", de_metric, prog_bar=True, logger=True)
        else:
            # For hyperspectral data, compute spectral angle mapper instead
            sam_metric = self._compute_spectral_angle_mapper(predictions, targets)
            self.log("test_sam", sam_metric, prog_bar=True, logger=True)

        self.log("test_psnr", psnr_metric, prog_bar=True, logger=True)  # Fixed typo
        self.log("test_ssim", ssim_metric, prog_bar=True, logger=True)
        self.log("test_loss", mae_loss, prog_bar=True, logger=True)
        return {"loss": mae_loss}

    def predict_step(self, batch, batch_idx):
        pathes, inputs = batch
        output = self(inputs)
        return output
