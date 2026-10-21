"""Read-only checks of split integrity, image paths and input preprocessing."""
import argparse
import json
from pathlib import Path

import torch

from .benchmarks import VTAB_CLASSES, VTAB_GROUPS
from .config import parse_args
from .data import DATASETS, FGVC, make_transform
from .vtab import VTAB


def check(dataset, data_root=None, split_root=None, verify_files=False):
    argv = ["--dataset", dataset]
    if data_root:
        argv += ["--data-root", data_root]
    if split_root:
        argv += ["--split-root", split_root]
    args = parse_args(argv)
    is_vtab = dataset in VTAB_CLASSES
    dataset_class = VTAB if is_vtab else FGVC
    train = dataset_class(args.data_root, dataset, "trainval",
                          make_transform(True, args.augmentation), args.seed, args.split_root)
    test = dataset_class(args.data_root, dataset, "test",
                         make_transform(False, "vtab" if is_vtab else "simple"),
                         split_root=args.split_root)
    if verify_files:
        missing = [str(path) for split in (train, test) for path, _ in split.samples if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing images; first: {missing[:3]}")
    for split in (train, test):
        image, _ = split[0]
        if image.shape != (3, 224, 224) or not torch.isfinite(image).all():
            raise ValueError(f"Invalid transformed image: {dataset}")
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(args.pretrained)
    return dict(dataset=dataset, suite="vtab" if is_vtab else "fgvc",
                group=next((key for key, values in VTAB_GROUPS.items() if dataset in values), None),
                classes=train.num_classes, train_samples=len(train), test_samples=len(test),
                image_root=str(train.image_root), split_root=str(train.split_root),
                pretrained=str(Path(args.pretrained).resolve()),
                all_image_paths_checked=verify_files, sample_shape=[3, 224, 224],
                train_manifest=train.manifest, test_manifest=test.manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["fgvc", "vtab", "all"], default="all")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS) + list(VTAB_CLASSES))
    parser.add_argument("--data-root")
    parser.add_argument("--split-root")
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    allowed = ((list(DATASETS) if args.suite in {"fgvc", "all"} else []) +
               (list(VTAB_CLASSES) if args.suite in {"vtab", "all"} else []))
    datasets = args.datasets or allowed
    if any(dataset not in allowed for dataset in datasets):
        parser.error("Requested dataset does not belong to --suite")
    torch.set_num_threads(2)
    for dataset in datasets:
        print(json.dumps(check(dataset, args.data_root, args.split_root, args.verify_files)), flush=True)


if __name__ == "__main__":
    main()
