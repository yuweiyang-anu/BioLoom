"""Run the BioLoomBench and BioMedFlickr protocols from the released data."""
from collections import defaultdict
import json
from pathlib import Path
import tarfile

import pyarrow.parquet as pq
import torch

from .common import batches

KS = (1, 10, 100)

def read_jsonl(path):
    with path.open() as file:
        return [json.loads(line) for line in file]

def encode_images(encoder, records, batch_size):
    return torch.cat([encoder.images(values).cpu() for values in batches(records, batch_size)])

def encode_texts(encoder, texts, batch_size, context=None):
    return torch.cat([encoder.texts(values, context).cpu() for values in batches(texts, batch_size)])

def recalls(queries, gallery, query_ids, gallery_ids, device, ks=KS):
    if len(queries) != len(query_ids) or len(gallery) != len(gallery_ids):
        raise ValueError("Feature count and relevance keys differ")
    members = defaultdict(list)
    for index, value in enumerate(gallery_ids):
        members[value].append(index)
    relevant = []
    for value in query_ids:
        keys = value if isinstance(value, set) else {value}
        relevant.append({i for key in keys for i in members.get(key, [])})
    counts = {k: 0 for k in ks}
    bank = gallery.to(device)
    for start in range(0, len(queries), 256):
        scores = queries[start:start + 256].to(device) @ bank.T
        ranks = scores.argsort(dim=1, descending=True, stable=True)[:, :max(ks)].cpu()
        for offset, retrieved in enumerate(ranks.tolist()):
            matches = relevant[start+offset]
            for k in ks:
                counts[k] += bool(matches.intersection(retrieved[:k]))
    return {f"R@{k}": 100 * count / len(queries) for k, count in counts.items()}

def bidirectional(images, texts, ids, device, subsets=None):
    def direction(rows):
        values = [ids[i] for i in rows]
        return {
            "image_to_text": recalls(images[rows], texts, values, ids, device),
            "text_to_image": recalls(texts[rows], images, values, ids, device),
        }
    result = direction(list(range(len(ids))))
    if subsets:
        result["by_query_subset"] = {
            field: {name: direction([i for i, value in enumerate(values) if value == name])
                    for name in sorted(set(values))}
            for field, values in subsets.items()
        }
    return result

def main_images(root, rows):
    wanted = [row["image_file"] for row in rows]
    cursor = 0
    for path in sorted((root / "main-data").glob("*.tar")):
        with tarfile.open(path) as archive:
            members = {member.name for member in archive if member.isfile()}
            while cursor < len(wanted) and wanted[cursor] in members:
                yield archive.extractfile(wanted[cursor]).read()
                cursor += 1
    if cursor != len(wanted):
        raise ValueError(f"Found {cursor} of {len(wanted)} BioLoomBench figures")

def bioloom(root, encoder, batch_size, groups=False):
    rows = read_jsonl(root / "metadata.jsonl")
    if len(rows) != 15_000:
        raise ValueError("Expected the released BioLoomBench gallery")
    images = encode_images(encoder, main_images(root, rows), batch_size)
    keys = [row["sample_id"] for row in rows]
    subsets = {k: [row[k] for row in rows] for k in ("benchmark_domain", "composition")} if groups else None
    tracks = (("concise", "concise_caption", None), ("original", "original_caption", None),
              ("original_cap77", "original_caption", min(77, encoder.tokenizer.context_length)))
    return {name: bidirectional(images, encode_texts(encoder, [r[field] for r in rows], batch_size, context),
                                keys, encoder.device, subsets) for name, field, context in tracks}

def flickr(root, encoder, batch_size):
    captions = []
    def images():
        for path in sorted((root / "data").glob("*.parquet")):
            for array in pq.ParquetFile(path).iter_batches(columns=["image", "caption"], batch_size=batch_size):
                for row in array.to_pylist():
                    captions.append(row["caption"])
                    yield row["image"]
    visuals = encode_images(encoder, images(), batch_size)
    if len(captions) != 7185:
        raise ValueError("Expected all 7,185 BioMedFlickr pairs")
    texts = encode_texts(encoder, captions, batch_size)
    return {name: bidirectional(visuals, texts, keys, encoder.device)
            for name, keys in (("same_caption", [" ".join(x.split()) for x in captions]),
                               ("exact_pair", list(range(len(captions)))))}

def panels(root, encoder, batch_size):
    from contextlib import ExitStack
    folder = root / "panel-retrieval"
    rows = [row for row in read_jsonl(folder / "metadata.jsonl") if row["split"] == "test"]
    whole = read_jsonl(folder / "whole_gallery.jsonl")
    if len(rows) != 21_975 or len(whole) != 15_000:
        raise ValueError("Missing evaluation images")
    def image_rows():
        last = None
        with ExitStack() as stack:
            for row in rows:
                if row["shard"] != last:
                    stack.close()
                    last = row["shard"]
                    archive = stack.enter_context(tarfile.open(folder / last))
                yield archive.extractfile(row["image_file"]).read()
    pi = encode_images(encoder, image_rows(), batch_size)
    pt = encode_texts(encoder, [r["caption"] for r in rows], batch_size)
    wi = encode_images(encoder, main_images(root, whole), batch_size)
    wt = encode_texts(encoder, [r["source_caption"] for r in whole], batch_size)
    norm = lambda text: " ".join(text.split()).casefold()
    caption_keys = [norm(r["caption"]) for r in rows]
    panel_keys = [r["pixel_sha256"] for r in rows]
    whole_image_keys = [r["image_sha256"] for r in whole]
    whole_caption_keys = [norm(r["source_caption"]) for r in whole]
    parent = {r["sample_id"]: i for i, r in enumerate(whole)}
    parent_rows = [parent[r["parent_sample_id"]] for r in rows]
    eligible = [i for i,r in enumerate(rows) if r["source_query_eligible"]]
    if len(eligible) != 19_622:
        raise ValueError("Wrong number of panel-to-caption queries")
    identical_panels = defaultdict(list)
    identical_captions = defaultdict(list)
    for i, (image_key, caption_key) in enumerate(zip(panel_keys, caption_keys)):
        identical_panels[image_key].append(i)
        identical_captions[caption_key].append(i)
    image_to_caption = [{caption_keys[j] for j in identical_panels[key]} for key in panel_keys]
    caption_to_image = [{panel_keys[j] for j in identical_captions[key]} for key in caption_keys]
    caption_to_parent = [
        {whole_image_keys[parent_rows[k]] for j in identical_captions[key] for k in identical_panels[panel_keys[j]]}
        for key in caption_keys
    ]
    image_to_source = [
        {whole_caption_keys[parent_rows[j]] for j in identical_panels[panel_keys[i]] if rows[j]["source_query_eligible"]}
        for i in eligible
    ]
    ks = (1, 5, 10, 100)
    return {
        "panel_to_panel_caption": recalls(pi, pt, image_to_caption, caption_keys, encoder.device, ks),
        "panel_caption_to_panel": recalls(pt, pi, caption_to_image, panel_keys, encoder.device, ks),
        "panel_caption_to_whole": recalls(pt, wi, caption_to_parent, whole_image_keys, encoder.device, ks),
        "panel_to_whole_caption": recalls(pi[eligible], wt, image_to_source, whole_caption_keys, encoder.device, ks),
    }
