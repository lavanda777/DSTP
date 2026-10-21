"""Stock timm ViT, loaded from a local NPZ; no import of the GPS repository."""

from pathlib import Path

import timm


def build_model(pretrained, num_classes, drop_path=0.0):
    path = Path(pretrained).expanduser().resolve()
    if path.suffix != ".npz" or not path.is_file():
        raise ValueError(f"A real local ViT-B/16 ImageNet-21K .npz is required: {path}")
    model = timm.create_model("vit_base_patch16_224_in21k", pretrained=False,
                              num_classes=num_classes, drop_path_rate=drop_path)
    # timm's NPZ loader skips the mismatched 21843-class head and loads the backbone.
    model.load_pretrained(str(path))
    if model.head.out_features != num_classes:
        raise RuntimeError("Incorrect target classification head")
    return model
