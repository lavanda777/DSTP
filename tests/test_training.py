import contextlib
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from timm.models.vision_transformer import VisionTransformer

from dstp.data import DATASETS
from dstp.train import main
from dstp.vtab import VTAB


def tiny_model(pretrained=None, num_classes=2, drop_path=0.0):
    return VisionTransformer(img_size=224, patch_size=16, embed_dim=24, depth=2,
                             num_heads=3, mlp_ratio=2, num_classes=num_classes,
                             drop_path_rate=drop_path)


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertEqual(left.dtype, right.dtype)
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tearDown(self):
        for handler in list(logging.getLogger().handlers):
            handler.close()
            logging.getLogger().removeHandler(handler)

    def run_quietly(self, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            main(argv)

    def fixture(self, root):
        images = root / "data/CUB_200_2011/images"
        images.mkdir(parents=True)
        rng = np.random.default_rng(45)
        for split, size in (("train", 8), ("val", 2), ("test", 2)):
            entries = {}
            for index in range(size):
                name = f"{split}_{index}.png"
                Image.fromarray(rng.integers(0, 256, (244, 252, 3), dtype=np.uint8)).save(images / name)
                entries[name] = index % 2
            (images.parent / f"{split}.json").write_text(json.dumps(entries))
        weights = root / "fixture.npz"
        weights.write_bytes(b"local fixture; model builder is patched")
        return ["--dataset", "cub", "--data-root", str(root / "data"),
                "--pretrained", str(weights), "--device", "cpu", "--amp", "none",
                "--workers", "0", "--batch-size", "4", "--epochs", "3", "--warmup-epochs", "0",
                "--ema-decay", "0.9", "--drop-path", "0.1"]

    def test_resume_matches_full_training_and_evaluation(self):
        # Exercise real AdamW, augmentation, Mixup, EMA, shuffling and checkpoint I/O.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            common = self.fixture(root)
            full, resumed = root / "full", root / "resumed"
            with patch.dict(DATASETS, {"cub": ("CUB_200_2011", "images", 2)}), patch(
                    "dstp.train.build_model", side_effect=tiny_model):
                self.run_quietly(common + ["--output", str(full)])
                self.run_quietly(common + ["--output", str(resumed), "--stop-after-steps", "1"])
                self.run_quietly(common + ["--resume", str(resumed / "last.pt")])
            left = torch.load(full / "last.pt", weights_only=False)
            right = torch.load(resumed / "last.pt", weights_only=False)
            for key in ("model", "optimizer", "scheduler", "scaler", "ema", "progress"):
                assert_tree_equal(self, left[key], right[key])
            self.assertEqual(left["scheduler"]["num_updates"], 6)
            self.assertEqual(left["progress"]["epoch"], 3)
            for a, b in zip(left["history"], right["history"]):
                for model_key in ("test", "test_ema"):
                    for metric in ("loss", "top1", "samples"):
                        self.assertEqual(a[model_key][metric], b[model_key][metric])
            best_left = torch.load(full / "best.pt", weights_only=False)
            best_right = torch.load(resumed / "best.pt", weights_only=False)
            for key in ("model", "optimizer", "scheduler", "ema", "progress"):
                assert_tree_equal(self, best_left[key], best_right[key])

    def test_changed_split_is_rejected_before_restoring_optimizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            common = self.fixture(root)
            output = root / "run"
            with patch.dict(DATASETS, {"cub": ("CUB_200_2011", "images", 2)}), patch(
                    "dstp.train.build_model", side_effect=tiny_model):
                self.run_quietly(common + ["--output", str(output), "--stop-after-steps", "1"])
                path = root / "data/CUB_200_2011/test.json"
                path.write_text(path.read_text() + "\n")
                with patch.object(torch.optim.AdamW, "load_state_dict") as restore:
                    with self.assertRaisesRegex(ValueError, "Evaluation/test data changed"):
                        self.run_quietly(common + ["--resume", str(output / "last.pt")])
                    restore.assert_not_called()

    def test_vtab_resume_without_ema_and_with_separate_split_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images/caltech101"
            splits = root / "splits/caltech101"
            images.mkdir(parents=True)
            splits.mkdir(parents=True)
            image = Image.new("RGB", (39, 77), (30, 180, 220))
            image.save(images / "train.png")
            image.save(images / "test.png")
            # Distinct metadata paths; decoder is patched to avoid 1000 image files.
            (splits / "train800val200.txt").write_text(
                "".join(f"train_{i}.png {i % 2}\n" for i in range(1000)))
            (splits / "test.txt").write_text("test.png 101\n")
            weights = root / "fixture.npz"
            weights.write_bytes(b"patched builder")
            common = ["--dataset", "caltech101", "--data-root", str(root / "images"),
                      "--split-root", str(root / "splits"), "--pretrained", str(weights),
                      "--device", "cpu", "--amp", "none", "--workers", "0", "--batch-size", "2",
                      "--epochs", "2", "--warmup-epochs", "0", "--stop-after-steps", "3"]
            original_getitem = VTAB.__getitem__

            def decode_fixture(dataset, index):
                if dataset.training:
                    return dataset.transform(image), dataset.samples[index][1]
                return original_getitem(dataset, index)

            with patch("dstp.train.build_model", side_effect=tiny_model), patch.object(
                    VTAB, "__getitem__", decode_fixture):
                self.run_quietly(common + ["--output", str(root / "full")])
                self.run_quietly(common + ["--output", str(root / "resumed"), "--stop-after-steps", "1"])
                self.run_quietly(common + ["--resume", str(root / "resumed/last.pt")])
            left = torch.load(root / "full/last.pt", weights_only=False)
            right = torch.load(root / "resumed/last.pt", weights_only=False)
            for key in ("model", "optimizer", "scheduler", "scaler", "ema", "progress"):
                assert_tree_equal(self, left[key], right[key])
            self.assertEqual(left["scheduler"]["num_updates"], 3)
            self.assertIsNone(left["ema"])

    def test_vtab_native_directories_and_alias_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, alias, classes in (("cifar100", "cifar", 100),
                                         ("flowers102", "oxford_flowers102", 102),
                                         ("pets", "oxford_iiit_pet", 37)):
                for directory in (name, alias):
                    folder = root / directory
                    folder.mkdir()
                    (folder / "train800val200.txt").write_text(
                        "".join(f"train_{i}.png {i % classes}\n" for i in range(1000)))
                    (folder / "test.txt").write_text("test.png 0\n")
                # Prefer native names when both directories exist.
                self.assertEqual(VTAB(root, name, "trainval", None).root, root / name)
                (root / name).rename(root / (name + "_hidden"))
                self.assertEqual(VTAB(root, name, "trainval", None).root, root / alias)


if __name__ == "__main__":
    unittest.main()
