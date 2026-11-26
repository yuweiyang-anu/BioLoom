"""Launch a BioLoom recipe on one multi-GPU node."""
import argparse
from collections import defaultdict
import re

from braceexpand import braceexpand
import os
from pathlib import Path
import shlex
import subprocess
import sys

from recipes import MODELS, MODEL_PREFIX

def shard_spec(paths):
    groups = defaultdict(list)
    plain = []
    for path in paths:
        match = re.fullmatch(r"(.*-)([0-9]+)\.tar", str(path))
        if match:
            groups[(match[1], len(match[2]))].append(int(match[2]))
        else:
            plain.append(str(path))
    spans = []
    for (prefix, width), numbers in groups.items():
        start = end = numbers[0]
        for number in [*numbers[1:], None]:
            if number == end + 1:
                end = number
                continue
            suffix = f"{start:0{width}d}" if start == end else f"{{{start:0{width}d}..{end:0{width}d}}}"
            spans.append(prefix + suffix + ".tar")
            start = end = number
    components = sorted(plain + spans, key=lambda component: next(braceexpand(component)))
    assert [str(path) for path in paths] == [f for c in components for f in braceexpand(c)]
    return "::".join(components)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MODELS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--samples", type=int, default=52_437_129)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    shards = sorted(args.data.resolve().glob("*.tar"))
    if not shards:
        parser.error("--data must contain Loom52M tar shards")
    if args.variant == "commonpool" and not args.vision_checkpoint:
        parser.error("CommonPool requires a source vision checkpoint via --vision-checkpoint")
    if args.vision_checkpoint and not args.vision_checkpoint.is_file():
        parser.error("Vision checkpoint does not exist")
    if args.gpus < 1 or args.batch_size < 1 or args.epochs < 1:
        parser.error("GPUs, batch size and epochs must be positive")
    if args.samples == 52_437_129 and len(shards) != 4525:
        parser.error("Full corpus expects 4,525 shards; use --samples for a smaller local run")
    learning_rate = "1e-4" if args.variant in {"dfn2b", "commonpool", "dinov2"} else "5e-4"
    arch = MODEL_PREFIX + MODELS[args.variant][2]
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.gpus),
        "-m", "--", "open_clip_train.main", "--name", args.variant, "--logs", str(args.output),
        "--train-data", shard_spec(shards), "--train-num-samples", str(args.samples),
        "--dataset-type", "webdataset", "--model", arch, "--lr", learning_rate,
        "--beta1", "0.9", "--beta2", "0.95", "--warmup", "2000", "--wd", "0.2",
        "--batch-size", str(args.batch_size), "--epochs", str(args.epochs), "--workers", str(args.workers),
        "--precision", "amp", "--local-loss", "--gather-with-grad", "--grad-clip-norm", "1.0",
        "--bioloom-loss", "--bio-mask-same-article-in-clip",
        "--bio-relation-weight", "0.05", "--bio-same-type-negative-weight", "1.1",
        "--bio-relation-warmup-steps", "34144", "--bio-type-balance-power", "0.25",
        "--bio-group-max-samples", "3", "--bio-group-shuffle-size", "512",
        "--bio-type-aware-groups", "--bio-rotate-panels", "--bio-group-balance-power", "0.5",
        "--bio-group-balance-window", "3072", "--bio-group-balance-candidates", "64",
        "--wds-shard-shuffle-size", "20000", "--wds-shard-shuffle-initial", "5000",
        "--seed", "0", "--save-frequency", "1", "--save-most-recent", "--save-steps-frequency", "5000",
        "--save-steps-max-keep", "10", "--report-to", "",
    ]
    if args.variant in {"dfn2b", "dinov2"} and not args.vision_checkpoint:
        command.append("--pretrained-image")
    if args.vision_checkpoint:
        command.extend(["--pretrained-image-path", str(args.vision_checkpoint)])
    if args.resume:
        command.extend(["--resume", str(args.resume)])
    if args.dry_run:
        printable = ["<ordered tar shards>" if arg.startswith(str(args.data.resolve())) else arg for arg in command]
        print(shlex.join(printable))
        return
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source, env.get("PYTHONPATH")]))
    subprocess.run(command, env=env, check=True)

if __name__ == "__main__":
    main()
