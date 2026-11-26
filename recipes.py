"""Released BioLoom model families."""
MODELS = {
    "scratch": ("Main/Scratch", "open_clip_pytorch_model.bin", "L-14-randvision"),
    "dfn2b": ("Main/DFN-2B", "open_clip_pytorch_model.bin", "L-14-pretrained"),
    "commonpool": ("Initialization/CommonPool", "open_clip_pytorch_model.bin", "L-14-commonpool-vision-longtext"),
    "dinov2": ("Initialization/DINOv2", "open_clip_pytorch_model.bin", "L-14-dinov2-vision-longtext"),
    "vit-s": ("Model-size/ViT-S-14", "open_clip_pytorch_model.bin", "S-14-randvision"),
    "vit-b": ("Model-size/ViT-B-14", "open_clip_pytorch_model.bin", "B-14-randvision"),
    "vit-h": ("Model-size/ViT-H-14", "open_clip_pytorch_model.bin", "H-14-randvision"),
}
MODEL_PREFIX = "BioClinicalModernBERT-large-ViT-"
MODELS_REPO = "ChartFoundation/BioLoom-models"
TRAIN_REPO = "ChartFoundation/Loom52M"
EVAL_REPO = "ChartFoundation/BioLoom-EvalSuite"
