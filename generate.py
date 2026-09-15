"""Inference pipeline: load a trained GPT checkpoint and generate text from a prompt."""

import argparse
import os

import tiktoken
import torch

from llm_architecture import (
    GPT_CONFIG_124M,
    GPTModel,
    generate_text_simple,
    load_model,
    text_to_token_ids,
    token_ids_to_text,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "model.pth")


def generate(
    prompt,
    model_path=MODEL_PATH,
    config=GPT_CONFIG_124M,
    max_new_tokens=50,
    use_trained=True,
):
    """Generate text from ``prompt`` using either a saved checkpoint or a random model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_trained:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No checkpoint found at {model_path}. Run `python train.py` first "
                "or pass --no-use-trained to initialize a random model."
            )
        model = load_model(config, model_path, device=device)
    else:
        model = GPTModel(config).to(device)

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
        "--use-trained",
        dest="use_trained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the saved checkpoint. Use --no-use-trained for a random GPTModel.",
    )
    args = parser.parse_args()

    output = generate(
        args.prompt,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        use_trained=args.use_trained,
    )
    print(output)


if __name__ == "__main__":
    main()
