import json
import os
import re

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

from encoder import get_encoder

def print_dict_with_array_sizes(d, indent=0, list_has_similar_elements=True):
    """For PRINT purposes only.
    list_has_similar_elements=True means that all elements in the list are of the same type, so print just the first.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print(' ' * indent + f"{key}:")
            print_dict_with_array_sizes(value, indent + 2)
        elif isinstance(value, list):
            nvalues = len(value)
            print(' ' * indent + f"{key}: [")
            for item in value:
                if isinstance(item, dict):
                    print(' ' * (indent + 2) + "---")
                    print_dict_with_array_sizes(item, indent + 2)
                    if list_has_similar_elements:
                        print(' ' * (indent + 2) + f"--- (other {nvalues} items of same type and size)")
                        break
                else:
                    print(' ' * (indent + 2) + str(item))
            print(' ' * indent + "]")
        elif isinstance(value, np.ndarray):
            print(' ' * indent + f"{key}: array_shape{value.shape}")
        else:
            print(' ' * indent + f"{key}: {value}")


def download_gpt2_files(model_size, model_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name[len("model/") :]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def load_encoder_hparams_and_params(model_size, models_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]

    model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    encoder = get_encoder(model_size, models_dir)
    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

    print(f"Loaded {model_size} model from {os.path.join(model_dir)}")
    print("hyperparams:", hparams)
    print(print_dict_with_array_sizes(params, indent=2))

    return encoder, hparams, params
