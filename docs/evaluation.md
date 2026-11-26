# Evaluate BioLoom

The evaluator downloads the selected inference weights from [BioLoom-models](https://huggingface.co/ChartFoundation/BioLoom-models). Pass `--download` to fetch inputs from the public [evaluation suite](https://huggingface.co/datasets/ChartFoundation/BioLoom-EvalSuite), or pass `--data path/to/BioLoom-EvalSuite` for an existing checkout.

Each variant has its own directory containing `open_clip_pytorch_model.bin` and `open_clip_config.json`. Set the first argument of `Encoder` to the desired variant. Run the example from the checkout after following the [installation instructions](../README.md).

| Variant | Model folder |
|---|---|
| `scratch` | `Main/Scratch` |
| `dfn2b` | `Main/DFN-2B` |
| `commonpool` | `Initialization/CommonPool` |
| `dinov2` | `Initialization/DINOv2` |
| `vit-s` | `Model-size/ViT-S-14` |
| `vit-b` | `Model-size/ViT-B-14` |
| `vit-h` | `Model-size/ViT-H-14` |

Score an image against two prompts:

```python
import torch
from evaluation.common import Encoder

encoder = Encoder("dfn2b", "cuda" if torch.cuda.is_available() else "cpu")
images = encoder.images(["figure.jpg"])
texts = encoder.texts(["a microscopy image", "a radiograph"])
print((images @ texts.T).cpu())
```

```bash
python -m evaluation --model dfn2b --benchmark zero-shot --download \
  --output results/dfn2b-zero-shot.json

python -m evaluation --model scratch --benchmark biomedflickr --download \
  --output results/scratch-biomedflickr.json

python -m evaluation --model dfn2b --benchmark bioloombench --groups --download \
  --output results/dfn2b-bioloombench.json

python -m evaluation --model dfn2b --benchmark panel-retrieval --download \
  --output results/dfn2b-panels.json
```

Use `--dataset octmnist` with zero-shot classification to check a single source before starting the complete evaluation. Use `--device cpu` or a smaller `--batch-size` when checking a subset.

| Benchmark | Output |
|---|---|
| Zero-shot | Per-task and domain accuracy, macro-F1 and AUC on the 67-task manifest, on a 0–1 scale. Overall weights the ten domains equally. |
| BioLoomBench | Bidirectional R@1, R@10 and R@100 (%) for concise captions, original captions at native context, and original captions capped at 77 tokens. Optional domain and visual-content results restrict queries and keep the full gallery. |
| BioMedFlickr | Bidirectional R@1, R@10 and R@100 (%) under the same-caption and exact-pair protocols. Same-caption matching ignores differences in whitespace. |
| Panel retrieval | R@1, R@5, R@10 and R@100 (%) for the four panel-level and cross-level tasks in Appendix I. |

For fixed-class zero-shot tasks, Macro-F1 includes all defined classes. One-vs-rest AUC uses the classes that have both positive and negative test examples.

Use `--reference` to download the reported scores from BioLoom-models for comparison. This skips model loading and inference.

```bash
python -m evaluation --model dfn2b --benchmark zero-shot --reference \
  --output results/dfn2b-zero-shot-reference.json
```
