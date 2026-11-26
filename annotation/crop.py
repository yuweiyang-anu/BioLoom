"""Crop each annotated panel with the release coordinate convention."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image
from annotate import validate

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("figures", type=Path)
    cli.add_argument("annotations", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    with args.figures.open() as stream:
        figures = {row["id"]: row for row in map(json.loads, stream)}
    args.output.mkdir(parents=True, exist_ok=True)
    with args.annotations.open() as source, (args.output / "panels.jsonl").open("w") as index:
        for figure_no, annotation in enumerate(map(json.loads, source)):
            parent = figures[annotation["id"]]
            validate(annotation["regions"])
            with Image.open(args.figures.parent / parent["image"]) as image:
                width, height = image.size
                for panel_no, region in enumerate(annotation["regions"]):
                    y0, x0, y1, x1 = region["box_2d"]
                    box = (x0 * width // 1000, y0 * height // 1000,
                           math.ceil(x1 * width / 1000), math.ceil(y1 * height / 1000))
                    filename = f"{figure_no:08d}-{panel_no:03d}.png"
                    image.crop(box).convert("RGB").save(args.output / filename)
                    record = {"parent_id": annotation["id"], "image": filename, **region}
                    index.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
