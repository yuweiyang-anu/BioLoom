# Training recipes

Run from the repository root after setting up the [environment](../README.md#environment). Use the full `train/` split from [Loom52M](https://huggingface.co/datasets/ChartFoundation/Loom52M). The launcher verifies the full-corpus shard count and keeps figure groups within shards. Plan for about 2.42 TB of training data plus space for checkpoints.

<table>
  <thead>
    <tr><th scope="col">Variant</th><th scope="col">Vision encoder</th><th scope="col">Initialization</th><th scope="col">Learning rate</th></tr>
  </thead>
  <tbody>
    <tr><th colspan="4" scope="rowgroup" align="left">ViT-L/14 initialization</th></tr>
    <tr><td><code>scratch</code></td><td>ViT-L/14</td><td>Random</td><td align="right">5e-4</td></tr>
    <tr><td><code>dfn2b</code></td><td>ViT-L/14</td><td>DFN-2B</td><td align="right">1e-4</td></tr>
    <tr><td><code>dinov2</code></td><td>ViT-L/14</td><td>DINOv2</td><td align="right">1e-4</td></tr>
    <tr><td><code>commonpool</code></td><td>ViT-L/14</td><td>CommonPool XL</td><td align="right">1e-4</td></tr>
  </tbody>
  <tbody>
    <tr><th colspan="4" scope="rowgroup" align="left">Vision encoder size variants</th></tr>
    <tr><td><code>vit-s</code></td><td>ViT-S/14</td><td>Random</td><td align="right">5e-4</td></tr>
    <tr><td><code>vit-b</code></td><td>ViT-B/14</td><td>Random</td><td align="right">5e-4</td></tr>
    <tr><td><code>vit-h</code></td><td>ViT-H/14</td><td>Random</td><td align="right">5e-4</td></tr>
  </tbody>
</table>

All recipes use the pretrained BioClinical ModernBERT text encoder, a 512-token context, masked image–text contrastive loss, and a relation term for different samples from the same figure. The default relation-loss and same-content-type hard-negative weights are **0.05** and **1.1**, respectively.

```bash
# Choose one variant per invocation.
python -m training.train --variant vit-s --data data/Loom52M/train
python -m training.train --variant vit-b --data data/Loom52M/train
python -m training.train --variant vit-h --data data/Loom52M/train
python -m training.train --variant dinov2 --data data/Loom52M/train

python -m training.commonpool --output data/commonpool-visual.pt
python -m training.train --variant commonpool --data data/Loom52M/train \
  --vision-checkpoint data/commonpool-visual.pt
```

Add `--dry-run` to inspect a command, or `--resume` with a complete local training checkpoint. The files in the Hugging Face model repository contain weights for evaluation. The launcher writes its own training checkpoints under `runs/` and keeps the most recent ten step checkpoints.
