"""Evaluate any released BioLoom checkpoint."""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from recipes import EVAL_REPO, MODELS, MODELS_REPO
from .common import Encoder

PATTERNS = {
    "bioloombench": ["bioloom-retrieval/main-data/*.tar", "bioloom-retrieval/metadata.jsonl"],
    "biomedflickr": ["biomedflickr-retrieval/data/*.parquet", "biomedflickr-retrieval/protocol.json"],
    "panel-retrieval": ["bioloom-retrieval/main-data/*.tar", "bioloom-retrieval/metadata.jsonl",
                        "bioloom-retrieval/panel-retrieval/*"],
    "zero-shot": ["zero-shot/*"],
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="dfn2b")
    parser.add_argument("--benchmark", choices=PATTERNS, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--reference", action="store_true", help="Save the published result JSON")
    parser.add_argument("--dataset", help="Run only this dataset_id in the zero-shot manifest")
    parser.add_argument("--groups", action="store_true", help="Include BioLoomBench domain and content results")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.reference:
        names = {"zero-shot": "zero-shot", "bioloombench": "bioloom-retrieval", "biomedflickr": "biomedflickr-retrieval"}
        if args.benchmark not in names:
            parser.error("No reference result is packaged for this benchmark")
        folder = MODELS[args.model][0]
        source = Path(hf_hub_download(MODELS_REPO, f"{folder}/results/{names[args.benchmark]}.json"))
        args.output.write_bytes(source.read_bytes())
        return
    if args.dataset and args.benchmark != "zero-shot":
        parser.error("--dataset is supported only for zero-shot")
    root = args.data or Path("data/BioLoom-EvalSuite")
    if args.download:
        globs = PATTERNS[args.benchmark]
        if args.dataset:
            meta = ["zero-shot/task_manifest.csv", "zero-shot/source_registry.json",
                    "zero-shot/prompt_views.json", "zero-shot/protocols/biomedica/caption_templates.json"]
            snapshot_download(EVAL_REPO, repo_type="dataset", local_dir=root, allow_patterns=meta)
            index = json.loads((root / "zero-shot/source_registry.json").read_text())["task_inputs"]
            values = {row["path"] for row in index if row["dataset_id"] == args.dataset}
            if not values:
                parser.error("Dataset ID not present")
            globs = [*meta, *(g for path in values for g in [path, f"{path}/*"])]
        snapshot_download(EVAL_REPO, repo_type="dataset", local_dir=root, allow_patterns=globs)
    encoder = Encoder(args.model, args.device, args.checkpoint)
    if args.benchmark == "zero-shot":
        from .classification import classify
        result = classify(root, encoder, args.batch_size, args.dataset)
    else:
        from .retrieval import bioloom, flickr, panels
        if args.benchmark == "bioloombench":
            result = bioloom(root / "bioloom-retrieval", encoder, args.batch_size, args.groups)
        elif args.benchmark == "biomedflickr":
            result = flickr(root / "biomedflickr-retrieval", encoder, args.batch_size)
        else:
            result = panels(root / "bioloom-retrieval", encoder, args.batch_size)
    args.output.write_text(json.dumps({"model": args.model, "benchmark": args.benchmark, **result},
                                      indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)

if __name__ == "__main__":
    main()
