"""
Contrastive Learning Model — InfoNCE / CLIP-style.

Two-tower projection network that aligns DINOv2 image embeddings (384-d)
and Nomic text embeddings (768-d) into a shared 512-d latent space.

Usage:
    .venv/bin/python src/lit_model.py [--epochs 50] [--batch-size 256] [--lr 1e-3]
"""

import argparse
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

DATA_DIR = Path("data")


class MetPairDataset(Dataset):
    """Dataset of (image_embedding, text_embedding) pairs."""

    def __init__(self, image_path: Path, text_path: Path, mask_path: Path | None = None):
        self.images = torch.load(image_path, map_location="cpu")
        self.texts = torch.load(text_path, map_location="cpu")
        assert self.images.shape[0] == self.texts.shape[0], \
            f"Mismatch: {self.images.shape[0]} images vs {self.texts.shape[0]} texts"

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx]


class MetDataModule(L.LightningDataModule):
    """Lightning DataModule for Met image-text pairs."""

    def __init__(self, data_dir: Path = DATA_DIR, batch_size: int = 256, val_split: float = 0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage=None):
        dataset = MetPairDataset(
            self.data_dir / "images_unprojected.pt",
            self.data_dir / "text_unprojected.pt",
        )
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        self.train_ds, self.val_ds = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)


class ContrastiveModel(L.LightningModule):
    """
    Two-tower contrastive model.
    Projects image (384-d) and text (768-d) into a shared 512-d space
    using InfoNCE loss with a learnable temperature.
    """

    def __init__(self, d_image: int = 384, d_text: int = 768, d_joint: int = 512, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.image_projector = nn.Sequential(
            nn.Linear(d_image, d_joint),
            nn.BatchNorm1d(d_joint),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_joint, d_joint),
        )
        self.text_projector = nn.Sequential(
            nn.Linear(d_text, d_joint),
            nn.BatchNorm1d(d_joint),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_joint, d_joint),
        )

        # Learnable log-temperature (initialized to ln(1/0.07) ≈ 2.66)
        self.log_temperature = nn.Parameter(torch.tensor(2.6593))

    def forward(self, images, texts):
        img_emb = F.normalize(self.image_projector(images), p=2, dim=1)
        txt_emb = F.normalize(self.text_projector(texts), p=2, dim=1)
        return img_emb, txt_emb

    def info_nce_loss(self, img_emb, txt_emb):
        """Symmetric InfoNCE contrastive loss."""
        temperature = self.log_temperature.exp()
        # Cosine similarity matrix (since embeddings are L2-normalized)
        logits = (img_emb @ txt_emb.T) * temperature
        targets = torch.arange(logits.shape[0], device=logits.device)

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        return (loss_i2t + loss_t2i) / 2

    def training_step(self, batch, batch_idx):
        images, texts = batch
        img_emb, txt_emb = self(images, texts)
        loss = self.info_nce_loss(img_emb, txt_emb)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("temperature", self.log_temperature.exp(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        img_emb, txt_emb = self(images, texts)
        loss = self.info_nce_loss(img_emb, txt_emb)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)


def main(epochs: int = 50, batch_size: int = 512, lr: float = 5e-4):
    dm = MetDataModule(batch_size=batch_size)
    model = ContrastiveModel(lr=lr)

    from lightning.pytorch.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.001)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        default_root_dir="checkpoints",
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=[early_stop]
    )

    trainer.fit(model, dm)

    # Save the final model
    ckpt_path = Path("checkpoints") / "contrastive_final.ckpt"
    trainer.save_checkpoint(ckpt_path)
    print(f"\n✓ Model saved to {ckpt_path}")

    # Project all embeddings and save for FAISS indexing
    dm.setup()
    model.eval()
    device = next(model.parameters()).device

    all_images = torch.load(DATA_DIR / "images_unprojected.pt", map_location="cpu")
    all_texts = torch.load(DATA_DIR / "text_unprojected.pt", map_location="cpu")

    with torch.no_grad():
        img_projected = F.normalize(model.image_projector(all_images.to(device)), p=2, dim=1).cpu()
        txt_projected = F.normalize(model.text_projector(all_texts.to(device)), p=2, dim=1).cpu()

    torch.save(img_projected, DATA_DIR / "images_projected.pt")
    torch.save(txt_projected, DATA_DIR / "text_projected.pt")
    print(f"✓ Projected embeddings saved: images={img_projected.shape}, texts={txt_projected.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train contrastive model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
