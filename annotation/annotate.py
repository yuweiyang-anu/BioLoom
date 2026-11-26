"""Annotate whole figures with Gemini 3.5 Flash-Lite and the Loom52M prompt."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import itertools
import mimetypes
import os
from pathlib import Path
import threading
import time

from schema import CONTENT_TYPE_ENUM, MODALITY_ENUM, RESPONSE_SCHEMA, build_prompt

def validate(regions):
    if not isinstance(regions, list) or not regions:
        raise ValueError("Expected one or more regions")
    ids = set()
    for region in regions:
        y0, x0, y1, x1 = region["box_2d"]
        if any(type(p) is not int for p in (y0, x0, y1, x1)) or not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
            raise ValueError("Invalid normalized box")
        if region["content_type"] not in CONTENT_TYPE_ENUM or region["modality"] not in MODALITY_ENUM:
            raise ValueError("Unexpected label")
        if any(type(region["layout"][k]) is not int or region["layout"][k] < 1 for k in ("rows", "cols")):
            raise ValueError("Invalid layout")
        if any(not isinstance(region[k], str) or not region[k].strip() for k in ("region_id", "data_description", "caption")):
            raise ValueError("Empty region field")
        if region["region_id"] in ids:
            raise ValueError("Repeated region identifier")
        ids.add(region["region_id"])

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--model", default="gemini-3.5-flash-lite")
    args = cli.parse_args()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        cli.error("Set GEMINI_API_KEY to your Google AI Studio API key.")
    from google import genai
    from google.genai import types
    tls = threading.local()
    def annotate(row):
        if not hasattr(tls, "client"):
            tls.client = genai.Client(api_key=api_key)
        path = args.input.parent / row["image"]
        mime = mimetypes.guess_type(path)[0]
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"Unsupported image type for {row['id']}")
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        )
        contents = [types.Part.from_bytes(data=path.read_bytes(), mime_type=mime),
                    build_prompt(row["caption"], row.get("mention_paragraphs", []))]
        for attempt in range(4):
            try:
                response = tls.client.models.generate_content(model=args.model, contents=contents, config=config)
                regions = json.loads(response.text)["regions"]
                validate(regions)
                return {"id": row["id"], "regions": regions}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {json.loads(line)["id"] for line in args.output.open()} if args.output.exists() else set()
    def records():
        with args.input.open() as source:
            for line in source:
                row = json.loads(line)
                if row["id"] not in completed:
                    completed.add(row["id"])
                    yield row
    with args.output.open("a") as output, ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(records())
        while batch := list(itertools.islice(iterator, args.workers * 4)):
            for result in pool.map(annotate, batch):
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()

if __name__ == "__main__":
    main()
