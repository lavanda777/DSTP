"""FGVC split integrity and reproducible augmentation; training uses train+val."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.transforms import RandomResizedCropAndInterpolation


DATASETS = {
    "cub": ("CUB_200_2011", "images", 200),
    "nabirds": ("nabirds", "images", 555),
    "flowers": ("oxfordflower", "", 102),
    "dogs": ("stanforddogs", "Images", 120),
    "cars": ("stanfordcars", "", 196),
}


def make_transform(training, augmentation="simple", crop_pct=0.9):
    interpolation = InterpolationMode.BICUBIC
    if augmentation == "vtab":
        # Local GPS passes input_size[-2:] (a tuple) to Resize in no-aug mode.
        return transforms.Compose([transforms.Resize((224, 224), interpolation=interpolation),
                                   transforms.CenterCrop((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    if training:
        ops = [RandomResizedCropAndInterpolation(224, scale=(0.08, 1.0),
                ratio=(0.75, 4 / 3), interpolation="random" if augmentation == "simple" else "bicubic"),
               transforms.RandomHorizontalFlip(0.5)]
    else:
        ops = [transforms.Resize(int(224 / crop_pct), interpolation=interpolation),
               transforms.CenterCrop(224)]
    return transforms.Compose(ops + [transforms.ToTensor(),
                                     transforms.Normalize((0.5,) * 3, (0.5,) * 3)])


class FGVC(Dataset):
    def __init__(self, root, dataset, split, transform, seed=3, split_root=None):
        if split not in {"train", "val", "trainval", "test"}:
            raise ValueError(split)
        directory, images, self.num_classes = DATASETS[dataset]
        self.root = Path(root).expanduser().resolve() / directory
        self.image_root = self.root / images
        self.split_root = Path(split_root).expanduser().resolve() / directory if split_root else self.root
        self.transform, self.seed, self.epoch = transform, seed, 0
        self.training = split in {"train", "trainval"}
        self.manifest = {}

        def read(part):
            path = self.split_root / f"{part}.json"
            raw = path.read_bytes()
            data = json.loads(raw)
            self.manifest[part] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                       count=len(data))
            return data

        train, val = read("train"), read("val")
        if set(train) & set(val):
            raise ValueError("train and val are not disjoint")
        labels = sorted(set(train.values()) | set(val.values()))
        if len(labels) != self.num_classes:
            raise ValueError(f"Class count mismatch for {dataset}: {len(labels)}")
        self.label_map = {label: i for i, label in enumerate(labels)}
        if split == "test":
            chosen = read("test")
            if (set(train) | set(val)) & set(chosen):
                raise ValueError("test overlaps training/validation")
        else:
            chosen = train if split == "train" else val if split == "val" else {**train, **val}
        self.samples = [(self.image_root / name, self.label_map[label])
                        for name, label in chosen.items()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if not self.training:
            return self.transform(image), target
        # Stable across workers and a mid-epoch restart. Augmentation
        # does not consume the model/optimizer RNG stream.
        seed = (self.seed * 1_000_003 + self.epoch * 1_000_000_007 + index) % (2 ** 32)
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                random.seed(seed)
                np.random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
                image = self.transform(image)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        return image, target


class EpochSampler(Sampler):
    def __init__(self, size, seed, epoch, start=0):
        self.size, self.seed, self.epoch, self.start = size, seed, epoch, start

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.size, generator=generator).tolist()[self.start:]

    def __len__(self):
        return self.size - self.start
