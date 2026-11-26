"""Extract the original CommonPool XL visual tower for initialization."""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    path = args.source or hf_hub_download("laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K", "open_clip_pytorch_model.bin")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k,v in checkpoint.get("state_dict", checkpoint).items()}
    tower = {k.removeprefix("visual."): v for k,v in state.items() if k.startswith("visual.")}
    if "conv1.weight" not in tower or "positional_embedding" not in tower:
        raise ValueError("Expected a full ViT-L/14 OpenCLIP checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tower,args.output)

if __name__ == "__main__":
    main()
