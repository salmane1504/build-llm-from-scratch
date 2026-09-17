"""Download and parse OpenAI's public GPT-2 checkpoints.

TensorFlow is imported lazily inside the functions so that the rest of the
package (and the checkpoint/random inference paths) do not require it.
"""

import json
import os

import numpy as np
import requests
from tqdm import tqdm

GPT2_ALLOWED_SIZES = ("124M", "355M", "774M", "1558M")
GPT2_FILENAMES = (
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
)
BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"
BACKUP_BASE_URL = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"


def download_and_load_gpt2(model_size, models_dir):
    """Download (if needed) and load the GPT-2 ``model_size`` weights.

    Returns a ``(settings, params)`` tuple where ``settings`` is the parsed
    ``hparams.json`` and ``params`` is a nested dict of numpy weight arrays.
    """
    import tensorflow as tf

    if model_size not in GPT2_ALLOWED_SIZES:
        raise ValueError(f"Model size not in {GPT2_ALLOWED_SIZES}")

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)
    for filename in GPT2_FILENAMES:
        file_url = os.path.join(BASE_URL, model_size, filename)
        backup_url = os.path.join(BACKUP_BASE_URL, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    with open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8") as f:
        settings = json.load(f)
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    """Stream ``url`` to ``destination``, falling back to ``backup_url``."""

    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024  # 1 KB
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """Read a GPT-2 TensorFlow checkpoint into a nested dict of numpy arrays."""
    import tensorflow as tf

    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
