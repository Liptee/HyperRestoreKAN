from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
import torch
import torchvision
import os
import torch.nn.functional as F


class GenerateCallback(Callback):
    def __init__(self, every_n_epochs=1) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.input_imgs = None
        self.save_dir = None
        self.target_imgs = None

    def hyperspectral_to_rgb(self, hyperspectral_img):
        """
        Convert 31-channel hyperspectral image to 3-channel RGB for visualization.
        Uses wavelength-to-RGB conversion based on typical hyperspectral band centers.
        
        Args:
            hyperspectral_img: Tensor of shape (B, 31, H, W)
        Returns:
            RGB tensor of shape (B, 3, H, W)
        """
        B, C, H, W = hyperspectral_img.shape
        
        if C != 31:
            # If not 31 channels, just take first 3 or duplicate single channel
            if C >= 3:
                return hyperspectral_img[:, :3, :, :]
            elif C == 1:
                return hyperspectral_img.repeat(1, 3, 1, 1)
            else:
                # Pad to 3 channels
                padding = torch.zeros(B, 3 - C, H, W, device=hyperspectral_img.device)
                return torch.cat([hyperspectral_img, padding], dim=1)
        
        # For 31-channel hyperspectral data, map to RGB
        # CAVE dataset typically covers 400-700nm range
        # Map bands to approximate RGB wavelengths
        # Red: bands 25-30 (around 650-700nm)  
        # Green: bands 12-17 (around 520-570nm)
        # Blue: bands 2-7 (around 420-470nm)
        
        red_channels = hyperspectral_img[:, 25:31, :, :].mean(dim=1, keepdim=True)
        green_channels = hyperspectral_img[:, 12:18, :, :].mean(dim=1, keepdim=True) 
        blue_channels = hyperspectral_img[:, 2:8, :, :].mean(dim=1, keepdim=True)
        
        rgb_img = torch.cat([red_channels, green_channels, blue_channels], dim=1)
        
        # Normalize to [0, 1] range
        rgb_img = torch.clamp(rgb_img, 0, 1)
        
        return rgb_img

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.val_dataloaders
        self.input_imgs, self.target_imgs = next(iter(dataloader))
        self.input_imgs = self.input_imgs.to(pl_module.device)
        self.target_imgs = self.target_imgs.to(pl_module.device)
        self.save_dir = os.path.join(trainer.log_dir, "figures")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(self.input_imgs)
                pl_module.train()
            
            # Convert hyperspectral images to RGB for visualization
            input_rgb = self.hyperspectral_to_rgb(self.input_imgs)
            target_rgb = self.hyperspectral_to_rgb(self.target_imgs)
            reconst_rgb = self.hyperspectral_to_rgb(reconst_imgs)
            
            # Plot and add to tensorboard
            imgs = torch.stack(
                [input_rgb, target_rgb, reconst_rgb], dim=1
            ).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3, normalize=True)
            # Save image
            save_path = os.path.join(
                self.save_dir, f"reconst_{trainer.current_epoch}.png"
            )
            os.makedirs(self.save_dir, exist_ok=True)
            torchvision.utils.save_image(grid, save_path)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.test_dataloaders
        self.input_imgs, self.target_imgs = next(iter(dataloader))
        self.input_imgs = self.input_imgs.to(pl_module.device)
        self.target_imgs = self.target_imgs.to(pl_module.device)
        self.save_dir = os.path.join(trainer.log_dir, "figures")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(self.input_imgs)
                pl_module.train()
            
            # Convert hyperspectral images to RGB for visualization
            input_rgb = self.hyperspectral_to_rgb(self.input_imgs)
            target_rgb = self.hyperspectral_to_rgb(self.target_imgs)
            reconst_rgb = self.hyperspectral_to_rgb(reconst_imgs)
            
            # Plot and add to tensorboard
            imgs = torch.stack(
                [input_rgb, target_rgb, reconst_rgb], dim=1
            ).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3, normalize=True)
            # Save image
            save_path = os.path.join(self.save_dir, f"test_{trainer.current_epoch}.png")
            os.makedirs(self.save_dir, exist_ok=True)
            torchvision.utils.save_image(grid, save_path)
