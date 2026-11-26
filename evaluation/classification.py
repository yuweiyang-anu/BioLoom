"""Zero-shot classification on the released BioLoom tasks."""
from collections import defaultdict
from contextlib import ExitStack
from itertools import groupby
import csv
import io
import json
from pathlib import Path
from statistics import mean
import tarfile

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .common import batches

METRICS = ("accuracy", "macro_f1", "auc")


def keyed_tars(path):
    """Index released image members by basename, rejecting ambiguities."""
    result = {}
    for shard in sorted(path.glob("*.tar")):
        with tarfile.open(shard) as archive:
            for entry in archive:
                if entry.isfile():
                    key = Path(entry.name).stem
                    if key in result:
                        raise ValueError(f"Duplicate image ID: {key}")
                    result[key] = (shard, entry.name)
    return result


def load_tar_rows(path, samples):
    last = None
    with ExitStack() as stack:
        for sample in samples:
            shard, name = path / sample["archive"], sample["path"]
            if shard != last:
                stack.close()
                archive = stack.enter_context(tarfile.open(shard))
                last = shard
            with archive.extractfile(name) as image:
                yield image.read()


def parquet_rows(path):
    for file in sorted(path.glob("*.parquet")) if path.is_dir() else [path]:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=64):
            yield from batch.to_pylist()


def regular_cases(path, task, templates):
    dataset, case = task["dataset_id"], task["task_id"]
    for row in parquet_rows(path):
        if dataset == "lc25000":
            if row["subset"] != case.split("_")[0]:
                continue
            mapping = templates["lc25000"][row["subset"]]
            label = int(row["answer_idx"])
        elif dataset == "patchcamelyon":
            mapping = templates["patchcamelyon"]
            choices = [tuple(mapping[str(i)][key] for key in ("False", "True")) for i in range(2)]
            yield row["image"], list(zip(*choices)), int(str(row["label"]).lower() != "normal")
            continue
        elif dataset == "chexpert":
            mapping = templates["chexpert"]
            label = int(row["answer_idx"])
        else:
            if row["task"] != task["subset"]:
                continue
            mapping = templates["medmnist"][row["task"]]
            label = list(mapping["0"]).index(str(row["label"]))
        yield row["image"], list(zip(*(x.values() for x in (mapping["0"], mapping["1"])))), label


def arrow_cases(path, task):
    import pyarrow as pa
    source = task["dataset_id"]
    name = task["task_id"]
    coarse = source == "ubench_coarse_attributes"
    if source.startswith(("burgess_", "eulenberg_")):
        source += "_" + name
    key = name if coarse else "classification"
    for file in sorted(path.glob("*.arrow")):
        with pa.memory_map(str(file)) as stream:
            for batch in pa.ipc.open_stream(stream):
                slots = [batch.schema.get_field_index(v) for v in ("dataset", "split", "captions", "image")]
                datasets, splits, captions, image = (batch.column(i) for i in slots)
                for i, (dataset, split) in enumerate(zip(datasets.to_pylist(), splits.to_pylist())):
                    if split != "test" or (not coarse and dataset != source):
                        continue
                    views = captions[i].as_py() or {}
                    a, b = views.get(key+"_0"), views.get(key+"_1")
                    if not a or not b or not a["options"] or not b["options"]:
                        continue
                    if a["answer_idx"] != b["answer_idx"] or len(a["options"]) != len(b["options"]):
                        raise ValueError(f"Unaligned prompt views: {task['task_id']}")
                    yield image[i].as_py(), list(zip(a["options"], b["options"])), int(a["answer_idx"])


def auc_score(labels, values):
    labels = np.asarray(labels)
    if not labels.any() or labels.all():
        return None
    return float(roc_auc_score(labels, values))


def multiclass(encoder, samples, batch_size):
    labels, scores, vocabularies = [], [], []
    prototypes = {}
    for chunk in batches(samples, batch_size):
        features = encoder.images([x[0] for x in chunk])
        for feature, (_, prompts, label) in zip(features, chunk):
            key = tuple(map(tuple, prompts))
            if key not in prototypes:
                prototypes[key] = encoder.prototypes(key)
            scores.append((100 * feature @ prototypes[key].T).softmax(-1).cpu().numpy())
            labels.append(label)
            vocabularies.append(tuple(p[0] for p in prompts))
    if not labels:
        raise ValueError("No matching evaluation images")
    predictions = [int(x.argmax()) for x in scores]
    if len(set(vocabularies)) == 1:
        values = np.stack(scores)
        class_ids = range(values.shape[1])
        targets = np.asarray(labels)
        if values.shape[1] == 2:
            auc = auc_score(targets, values[:, 1])
        else:
            aucs = [auc_score(targets == i, values[:, i]) for i in class_ids]
            valid_aucs = [value for value in aucs if value is not None]
            auc = mean(valid_aucs) if valid_aucs else None
        truth, predicted = labels, predictions
    else:
        class_ids = None
        truth = [v[y] for v, y in zip(vocabularies, labels)]
        predicted = [v[y] for v, y in zip(vocabularies, predictions)]
        auc = auc_score([int(i == y) for y, v in zip(labels, scores) for i in range(len(v))],
                        np.concatenate(scores))
    return {"accuracy": float(np.mean(np.equal(labels, predictions))),
            "macro_f1": float(f1_score(truth, predicted, labels=class_ids, average="macro", zero_division=0)),
            "auc": auc, "n": len(labels)}


def multilabel(encoder, images_labels, prompts, batch_size, binary=False):
    levels = torch.stack([encoder.prototypes(p) for p in prompts])
    scores, labels = [], []
    for chunk in batches(images_labels, batch_size):
        features = encoder.images([row[0] for row in chunk])
        scores.extend((100 * torch.einsum("bd,lpd->blp", features, levels)).softmax(-1)[..., 1].cpu().numpy())
        labels.extend(row[1] for row in chunk)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predictions = scores >= 0.5
    aucs = [auc_score(labels[:, i], scores[:, i]) for i in range(labels.shape[1])]
    if binary:
        labels, predictions = labels[:, 0], predictions[:, 0]
    values = [x for x in aucs if x is not None]
    return {"accuracy": float((labels == predictions).mean()),
            "macro_f1": float(f1_score(labels, predictions, labels=[0, 1] if binary else None,
                                       average="macro", zero_division=0)),
            "auc": mean(values) if values else None, "n": len(labels)}


def chest_and_dresden(encoder, path, task, prompt_registry, batch_size):
    if task["dataset_id"] == "chestmnist":
        pairs = prompt_registry["chestmnist"]["labels"]
        with np.load(path) as data:
            return multilabel(encoder, zip(data["test_images"], data["test_labels"]),
                              [[v["negative"], v["positive"]] for v in pairs.values()], batch_size)
    if task["task_id"] == "multilabel_anatomy":
        pairs = prompt_registry["dresden_multilabel_anatomy"]["labels"]
        names = list(pairs)
        prompts = [[v["negative"], v["positive"]] for v in pairs.values()]
    else:
        name = task["task_id"].removesuffix("_presence").replace("_", " ")
        names = [name]
        prompts = [[[f"{name.capitalize()} is not visible in this image.", f"This image does not show the {name} clearly."],
                    [f"{name.capitalize()} is visible in this image.", f"This image shows the {name} clearly."]]]
    items = ((r["image"], [int(json.loads(r["metadata"])["anatomy_labels"][c]) for c in names])
             for r in parquet_rows(path))
    return multilabel(encoder, items, prompts, batch_size, binary=len(names) == 1)


def external_cases(path, key, views):
    import pandas as pd
    if key == "deepdrid":
        frame = pd.read_excel(path / "Challenge1_labels.xlsx")
        labels = zip(frame["image_id"].astype(str), frame["DR_Levels"].astype(int))
    elif key == "ham10000":
        frame = pd.read_csv(path / "ISIC2018_Task3_Test_GroundTruth.csv")
        labels = zip(frame["image"].astype(str), frame[["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]].to_numpy().argmax(1))
    else:
        frame = pd.read_csv(path / "stage_2_train_labels.csv").groupby("patientId", as_index=False)["Target"].max()
        labels = zip(frame["patientId"].astype(str), frame["Target"].astype(int))
    index = keyed_tars(path)
    cases = sorted((index[Path(name).stem], label) for name, label in labels)
    for shard, group in groupby(cases, key=lambda item: item[0][0]):
        with tarfile.open(shard) as archive:
            for (_, member), label in group:
                if member.lower().endswith(".dcm"):
                    import pydicom
                    pixels = pydicom.dcmread(io.BytesIO(archive.extractfile(member).read())).pixel_array.astype(np.float32)
                    low, high = np.percentile(pixels, [0.5, 99.5])
                    img = ((np.clip((pixels - low) / max(high - low, 1e-6), 0, 1)) * 255).astype(np.uint8)
                else:
                    img = archive.extractfile(member).read()
                yield img, list(zip(*views)), int(label)


def aggregate(rows):
    return {m: mean(vals) if (vals := [r[m] for r in rows if r[m] is not None]) else None for m in METRICS}


def classify(root, encoder, batch_size, dataset=None):
    with (root / "zero-shot/task_manifest.csv").open() as file:
        rows = list(csv.DictReader(file))
    index = json.loads((root / "zero-shot/source_registry.json").read_text())["task_inputs"]
    lookup = {(row["dataset_id"], row["task_id"]): row for row in index}
    prefix = root / "zero-shot"
    templates = json.loads((prefix / "protocols/biomedica/caption_templates.json").read_text())
    prompts = json.loads((prefix / "prompt_views.json").read_text())["tasks"]
    task_results = []
    for row in rows:
        name = row["dataset_id"]
        if dataset and name != dataset:
            continue
        task = lookup[name, row["task_id"]]
        path = root / task["path"]
        if "manifests" in task:
            paths = [root / p for p in task["manifests"] if name == "coos7" or Path(p).stem == row["task_id"]]
            if not paths and len(task["manifests"]) == 1:
                paths = [root / task["manifests"][0]]
            scores = []
            union_count = 0
            for manifest in paths:
                spec = json.loads(manifest.read_text())
                parts = defaultdict(list)
                union_count += len(spec["samples"])
                for sample in spec["samples"]:
                    partitions = sample["official_test_splits"] if name == "lids" else [None]
                    for part in partitions:
                        parts[part].append(sample)
                if name == "lids" and len(parts) != 3:
                    raise ValueError("Expected three publisher test splits")
                for samples in parts.values():
                    labels = (s["label"] for s in samples)
                    scores.append(multiclass(encoder, ((v, spec["prompts"], y)
                        for v, y in zip(load_tar_rows(path, samples), labels)), batch_size))
            result = {**aggregate(scores), "n": union_count}
        elif name in ("chestmnist", "dresden_surgical_anatomy"):
            result = chest_and_dresden(encoder, path, task, prompts, batch_size)
        else:
            source = row["source_key"]
            if source in ("raw_ubench", "biomedica_ubench"):
                cases = arrow_cases(path, task)
            elif name in ("deepdrid", "ham10000", "rsna_pneumonia"):
                cases = external_cases(path, name, prompts[name]["views"])
            else:
                cases = regular_cases(path, task, templates)
            result = multiclass(encoder, cases, batch_size)
        if result["n"] != int(row["sample_count"]):
            raise ValueError(f"{name}/{row['task_id']} expected {row['sample_count']} samples, got {result['n']}")
        task_results.append({"dataset_id": name, "domain": row["domain"], "task_id": row["task_id"], **result})
        print(f"{name}/{row['task_id']} accuracy={result['accuracy']:.4f}", flush=True)
    if not task_results:
        raise ValueError("Dataset is absent from the task manifest")
    groups = defaultdict(list)
    for task in task_results:
        groups[task["domain"], task["dataset_id"]].append(task)
    domains = defaultdict(list)
    for (domain, _), scores in groups.items():
        domains[domain].append(aggregate(scores))
    summaries = {k: aggregate(v) for k, v in domains.items()}
    return {"complete": len(task_results) == len(rows), "tasks": task_results, "domains": summaries,
            "overall": aggregate(list(summaries.values()))}
