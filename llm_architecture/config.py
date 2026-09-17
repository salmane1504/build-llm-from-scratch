"""Model and training hyperparameters."""

# GPT-2 small (124M) architecture configuration.
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,      # Query-key-value bias
}

# Optimizer / training loop settings.
TRAIN_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 10,
    "batch_size": 2,
    "weight_decay": 0.1,
}

# Per-size overrides for OpenAI's pretrained GPT-2 checkpoints.
GPT2_MODEL_CONFIGS = {
    "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def build_gpt2_config(model_size="124M"):
    """Return a ``GPTModel`` config matching OpenAI's pretrained GPT-2 weights.

    Uses the full 1024-token context length and enables the query-key-value bias
    that the original GPT-2 checkpoints were trained with.
    """
    if model_size not in GPT2_MODEL_CONFIGS:
        raise ValueError(f"model_size must be one of {tuple(GPT2_MODEL_CONFIGS)}")

    config = GPT_CONFIG_124M.copy()
    config.update(GPT2_MODEL_CONFIGS[model_size])
    config.update({"context_length": 1024, "qkv_bias": True})
    return config
