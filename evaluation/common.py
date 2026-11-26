"""Shared checkpoint loading and image/text feature encoding."""
import io
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from recipes import MODELS, MODELS_REPO, MODEL_PREFIX

def batches(values, batch_size):
    iterator = iter(values)
    while chunk := list(islice(iterator, batch_size)):
        yield chunk

def rgb(value):
    if isinstance(value, (str, Path)) and str(value).lower().endswith(".dcm"):
        import pydicom
        array = pydicom.dcmread(value).pixel_array.astype(np.float32)
        low, high = np.percentile(array, [0.5, 99.5])
        array = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
        return Image.fromarray((array * 255).astype(np.uint8)).convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")
    if isinstance(value, dict):
        value = value.get("bytes") or value["path"]
    if isinstance(value, (bytes, bytearray)):
        value = io.BytesIO(value)
    with Image.open(value) as image:
        return image.convert("RGB")

class Encoder:
    def __init__(self, variant, device, checkpoint=None):
        import open_clip
        folder, weights, arch = MODELS[variant]
        self.arch = MODEL_PREFIX + arch
        self.device = torch.device(device)
        if checkpoint is None:
            snapshot = snapshot_download(
                MODELS_REPO, allow_patterns=[f"{folder}/{weights}", f"{folder}/open_clip_config.json"]
            )
            self.model, self.preprocess = open_clip.create_model_from_pretrained(
                f"local-dir:{Path(snapshot) / folder}", device="cpu"
            )
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.arch, pretrained=None, pretrained_image=False, pretrained_text=False, device="cpu"
            )
            state = torch.load(checkpoint,
                               weights_only=True, map_location="cpu")
            self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.get("state_dict", state).items()},
                                       strict=True)
        self.model = self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.arch)

    def images(self, batch):
        tensors = torch.stack([self.preprocess(rgb(image)) for image in batch])
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            features = self.model.encode_image(tensors.to(self.device))
        return F.normalize(features.float(), dim=-1)

    def texts(self, batch, length=None):
        tokens = self.tokenizer(batch, context_length=length) if length else self.tokenizer(batch)
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            features = self.model.encode_text(tokens.to(self.device))
        return F.normalize(features.float(), dim=-1)

    def prototypes(self, classes):
        features = self.texts([prompt for prompts in classes for prompt in prompts])
        counts = [len(prompts) for prompts in classes]
        if len(set(counts)) == 1:
            means = features.reshape(len(classes), counts[0], -1).mean(1)
        else:
            means = torch.stack([group.mean(0) for group in features.split(counts)])
        return F.normalize(means, dim=-1)
