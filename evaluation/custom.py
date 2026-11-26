"""Run a released BioLoom model on a local classification or retrieval task."""
import argparse
import json
from pathlib import Path

from recipes import MODELS
from .common import Encoder, batches
from .classification import multiclass
from .retrieval import bidirectional, encode_images, encode_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="dfn2b", help="BioLoom checkpoint (default: %(default)s)")
    parser.add_argument("--mode", choices=("predict", "zero-shot", "retrieval"), default="zero-shot",
                        help="Predict labels, score labeled images, or score image–caption retrieval")
    parser.add_argument("--task", type=Path, required=True, help="Local task JSON")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        parser.error("The task JSON must be an object")
    samples = task.get("samples")
    if not isinstance(samples, list) or not samples or any(
        not isinstance(sample, dict) or not isinstance(sample.get("image"), str) or not sample["image"].strip()
        for sample in samples
    ):
        parser.error("Provide a non-empty 'samples' list with an image path in each sample")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    images = [args.task.parent / sample["image"] for sample in samples]
    if args.mode == "retrieval":
        captions = [sample.get("caption") for sample in samples]
        if any(not isinstance(caption, str) or not caption.strip() for caption in captions):
            parser.error("Provide a non-empty caption for every retrieval image")
        encoder = Encoder(args.model, args.device)
        visual = encode_images(encoder, images, args.batch_size)
        textual = encode_texts(encoder, captions, args.batch_size)
        scores = {"n": len(samples), **bidirectional(visual, textual, list(range(len(samples))), encoder.device)}
    else:
        scores = classify_task(parser, args, task.get("classes"), samples, images)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": args.model, "task": args.task.stem, "mode": args.mode, **scores}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


def classify_task(parser, args, classes, samples, images):
    if not isinstance(classes, dict) or len(classes) < 2:
        parser.error("Define at least two classes with a description for each")
    prompts = [[text] if isinstance(text, str) else text for text in classes.values()]
    if any(not isinstance(group, list) or not group
           or any(not isinstance(text, str) or not text.strip() for text in group)
           for group in prompts):
        parser.error("Class descriptions must contain non-empty text")
    if args.mode == "predict":
        encoder = Encoder(args.model, args.device)
        prototypes = encoder.prototypes(prompts)
        classes_in_order = list(classes)
        predictions = []
        for batch in batches(images, args.batch_size):
            indices = (encoder.images(batch) @ prototypes.T).argmax(dim=-1).tolist()
            predictions.extend(classes_in_order[index] for index in indices)
        return {"n": len(predictions), "predictions": [
            {"image": sample["image"], "predicted_label": label} for sample, label in zip(samples, predictions)
        ]}

    class_indices = {label: index for index, label in enumerate(classes)}
    if any(not isinstance(sample.get("label"), str) or sample["label"] not in class_indices for sample in samples):
        parser.error("Use a label from 'classes' for each image, or use --mode predict for unlabeled images")

    cases = ((image, prompts, class_indices[sample["label"]]) for image, sample in zip(images, samples))
    return multiclass(Encoder(args.model, args.device), cases, args.batch_size)


if __name__ == "__main__":
    main()
