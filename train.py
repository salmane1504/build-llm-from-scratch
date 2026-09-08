"""Training pipeline: prepare data, train the GPT model, and save the checkpoint."""

import os

import matplotlib.pyplot as plt
import requests
import tiktoken
import torch

from llm_architecture import (
    GPT_CONFIG_124M,
    GPTModel,
    TRAIN_SETTINGS,
    create_dataloader_v1,
    save_model,
)
from llm_architecture.training import plot_losses, train_model_simple

# Project paths.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "text", "the-verdict.txt")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model.pth")
LOSS_PLOT_PATH = os.path.join(CHECKPOINT_DIR, "loss.pdf")
DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch02/01_main-chapter-code/the-verdict.txt"
)


def load_text_data(file_path=DATA_PATH, url=DATA_URL):
    """Return the training text, downloading it once if it is missing."""
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def build_dataloaders(text_data, gpt_config, settings, train_ratio=0.90):
    """Split ``text_data`` and build train/validation dataloaders."""
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def train(gpt_config, settings):
    """Run the full training pipeline and return the trained model and metrics."""
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_data = load_text_data()
    train_loader, val_loader = build_dataloaders(text_data, gpt_config, settings)

    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer,
    )
    return model, train_losses, val_losses, tokens_seen


def main():
    model, train_losses, val_losses, tokens_seen = train(GPT_CONFIG_124M, TRAIN_SETTINGS)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Plot and persist the loss curves.
    epochs_tensor = torch.linspace(0, TRAIN_SETTINGS["num_epochs"], len(train_losses))
    fig = plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    fig.savefig(LOSS_PLOT_PATH)
    plt.close(fig)

    # Persist the trained model for the inference pipeline.
    save_model(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved loss plot to {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
