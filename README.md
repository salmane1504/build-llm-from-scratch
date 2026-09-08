# build-llm-from-scratch

A minimal GPT-style language model built from scratch with PyTorch, organized as a
reusable package plus separate **training** and **inference** pipelines.

## Project structure

```
build-llm-from-scratch/
├── train.py                  # Training pipeline entry point
├── generate.py               # Inference pipeline entry point (loads a trained model)
├── requirements.txt
├── llm_architecture/         # Isolated LLM code (importable package)
│   ├── config.py             # Model + training hyperparameters
│   ├── utils.py              # Model save/load helpers
│   ├── data/                 # Dataset + dataloader
│   ├── layers/               # GELU, FeedForward, LayerNorm
│   ├── models/               # GPTModel, MultiHeadAttention, TransformerBlock
│   ├── generation/           # Text generation utilities
│   └── training/             # Training loop, loss + evaluation utilities
├── checkpoints/              # Saved models & loss plots (created at train time)
├── notebooks/                # Exploratory notebooks
└── text/                     # Training corpus
```

All model code lives inside `llm_architecture`. The root-level `train.py` and
`generate.py` are thin pipelines that import from that package.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Downloads the corpus if needed, trains the model, and saves the checkpoint and
loss plot to `checkpoints/`:

```bash
python train.py
```

## Inference

Loads the trained checkpoint and continues text from a prompt:

```bash
python generate.py --prompt "Every effort moves you" --max-new-tokens 50
```
