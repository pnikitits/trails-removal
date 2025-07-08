import torch
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from dataset import FITSSatelliteTileDataset
from model import UNet
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from glob import glob
import random
import os
import time
import wandb
from utils import show, check_image_range, auto_stretch
from pathlib import Path

LOG_WANDB = False


def train_model(
    model: torch.nn.Module,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
    save_path: str = None,
    save_every: int = 5,
) -> tuple[torch.nn.Module, list[float], list[float], list[float], list[float]]:
    """
    Train the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    num_epochs : int
        Number of epochs to train for.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    criterion : torch.nn.Module
        Loss function to use.
    device : str
        Device to run the training on ('cpu', 'cuda', or 'mps').
    save_path : str, optional
        Path to save the model weights after training. If None, no saving is done.
    save_every : int, optional
        Save the model every n epochs. Default is 5.

    Returns
    -------
    tuple[torch.nn.Module, list[float], list[float], list[float], list[float]]
        The trained model, training losses, training x values (epochs),
        validation losses, and validation x values (epochs).
    """

    log_train_y = []  # losses
    log_train_x = []  # epochs

    log_val_y = []
    log_val_x = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="Training", ncols=100
        )

        for batch_idx, (noisy, clean) in progress_bar:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            # log losses
            log_train_y.append(loss.item())
            log_train_x.append(epoch + batch_idx / len(train_loader))

            avg_train_loss = epoch_train_loss / (batch_idx + 1)
            progress_bar.set_postfix(train_loss=avg_train_loss)

            if LOG_WANDB:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "epoch": epoch + batch_idx / len(train_loader),
                    }
                )

        # validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)

        # log losses
        log_val_y.append(avg_val_loss)
        log_val_x.append(epoch + 1)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1} complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.1f}s"
        )

        if save_path and (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), save_path / f"model_epoch_{epoch + 1}.pth")
            print(f"Model saved at {save_path}/model_epoch_{epoch + 1}.pth")

        if LOG_WANDB:
            wandb.log({"val/loss": avg_val_loss, "epoch": epoch + 1})

    return model, log_train_y, log_train_x, log_val_y, log_val_x


def check_sample(dataset: FITSSatelliteTileDataset, idx: int, verbose: bool = False):
    """
    Check a sample pair from the dataset.
    """
    sample_input, sample_target = dataset[idx]
    sample_input = sample_input.numpy()
    sample_target = sample_target.numpy()

    sample_input = np.transpose(sample_input, (1, 2, 0))
    sample_target = np.transpose(sample_target, (1, 2, 0))

    if verbose:
        print(f"Dataset size: {len(dataset)}")
        check_image_range(sample_input)
        check_image_range(sample_target)

    show(
        [sample_input, sample_target],
        title=["Noisy Input", "Clean Target"],
    )


def main():
    """
    Main function to train and save the model for satellite trail removal.
    """

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    full_dataset = FITSSatelliteTileDataset(
        directory=Path("data") / "debayered_set",
        tile_size=256,
        overlap=32,
        augment=True,
        preload=False,
    )

    # --- Check a sample of the dataset ---
    # for i in range(100):
    #     check_sample(full_dataset, i)
    # return
    # --- Check a sample of the dataset ---

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=3, persistent_workers=True
    )

    model = UNet().to(device)

    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = Path("results") / timestamp
    os.makedirs(save_path, exist_ok=True)

    model, log_train_y, log_train_x, log_val_y, log_val_x = train_model(
        model=model,
        num_epochs=20,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        criterion=torch.nn.MSELoss(),
        device=device,
        save_path=save_path,
        save_every=4,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(log_train_x, log_train_y, label="Training Loss")
    plt.plot(log_val_x, log_val_y, label="Validation Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / "loss_curves.png")
    plt.clf()

    model.eval()
    with torch.no_grad():
        sample_input, sample_target = next(iter(train_loader))
        sample_output = model(sample_input.to(device)).cpu().numpy()

    idx = 0
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_input[idx].cpu().numpy().squeeze(), cmap="gray")
    plt.title("Noisy Input")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(sample_target[idx].cpu().numpy().squeeze(), cmap="gray")
    plt.title("Clean Target")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(sample_output[idx].squeeze(), cmap="gray")
    plt.title("Model Output")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path / "sample_output.png")


if __name__ == "__main__":
    if LOG_WANDB:
        wandb.init(
            project="satellite-trail-removal",
            config={
                "epochs": 20,
                "batch_size": 8,
                "learning_rate": 1e-3,
                "loss_function": "MSELoss",
                "optimizer": "Adam",
                "model": "UNet",
            },
        )

    main()

    if LOG_WANDB:
        wandb.finish()
