"""Inference pipeline: generate text from a prompt using a GPT model.

The model weights can come from three sources:

* ``checkpoint`` - a locally trained checkpoint (default),
* ``random``     - a freshly initialized (untrained) ``GPTModel``,
* ``openai``     - OpenAI's pretrained GPT-2 weights, downloaded on demand.
"""

import argparse
import os

import tiktoken
import torch

from llm_architecture import (
    GPT_CONFIG_124M,
    GPT2_MODEL_CONFIGS,
    GPTModel,
    generate_text_simple,
    load_model,
    load_pretrained_gpt,
    text_to_token_ids,
    token_ids_to_text,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "model.pth")
GPT2_DIR = os.path.join(PROJECT_ROOT, "gpt2")


def generate(
    prompt,
    source="checkpoint",
    model_path=MODEL_PATH,
    config=GPT_CONFIG_124M,
    gpt2_size="124M",
    models_dir=GPT2_DIR,
    max_new_tokens=50,
):
    """Generate text continued from ``prompt``.

    ``source`` selects the weights: ``"checkpoint"`` loads a trained model,
    ``"random"`` uses an untrained ``GPTModel``, and ``"openai"`` downloads and
    loads OpenAI's pretrained GPT-2 ``gpt2_size`` weights.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if source == "checkpoint":
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No checkpoint found at {model_path}. Run `python train.py` first, "
                "or pass --source random / --source openai."
            )
        model = load_model(config, model_path, device=device)
    elif source == "random":
        model = GPTModel(config).to(device)
    elif source == "openai":
        model, config = load_pretrained_gpt(
            gpt2_size, models_dir=models_dir, device=device
        )
    else:
        raise ValueError(
            f"Unknown source {source!r}. Choose 'checkpoint', 'random', or 'openai'."
        )

    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    token_ids = generate_text_simple(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=config["context_length"],
    )
    return token_ids_to_text(token_ids, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Generate text with a GPT model.")
    parser.add_argument("--prompt", default="Every effort moves you", help="Starting text.")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Tokens to generate.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to the checkpoint.")
    parser.add_argument(
        "--source",
        choices=["checkpoint", "random", "openai"],
        default="checkpoint",
        help=(
            "Weights to use: 'checkpoint' (trained), 'random' (untrained), "
            "or 'openai' (pretrained GPT-2)."
        ),
    )
    parser.add_argument(
        "--gpt2-size",
        choices=list(GPT2_MODEL_CONFIGS),
        default="124M",
        help="Pretrained GPT-2 size to use when --source openai.",
    )
    parser.add_argument(
        "--models-dir",
        default=GPT2_DIR,
        help="Directory to cache downloaded OpenAI GPT-2 weights.",
    )
    args = parser.parse_args()

    output = generate(
        args.prompt,
        source=args.source,
        model_path=args.model_path,
        gpt2_size=args.gpt2_size,
        models_dir=args.models_dir,
        max_new_tokens=args.max_new_tokens,
    )
    print(output)


if __name__ == "__main__":
    main()
