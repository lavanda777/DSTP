"""GPS-style VTAB-1k image lists, with split integrity and fixed class identities."""
import hashlib
from pathlib import Path

from .benchmarks import VTAB_CLASSES, VTAB_DIRECTORIES
from .data import FGVC


class VTAB(FGVC):
    # Reuse only image decoding / sample-and-epoch augmentation RNG isolation.
    def __init__(self, root, dataset, split, transform, seed=42, split_root=None):
        if split not in {"trainval", "test"}:
            raise ValueError("VTAB uses train800val200.txt and test.txt")
        root = Path(root).expanduser().resolve()
        directory = dataset if (root / dataset).is_dir() else VTAB_DIRECTORIES.get(dataset, dataset)
        self.root = (root / directory).resolve()
        if split_root:
            split_root = Path(split_root).expanduser().resolve()
            split_directory = dataset if (split_root / dataset).is_dir() else VTAB_DIRECTORIES.get(dataset, dataset)
            self.split_root = split_root / split_directory
        else:
            self.split_root = self.root
        self.image_root = self.root
        self.num_classes = VTAB_CLASSES[dataset]
        self.transform, self.seed, self.epoch = transform, seed, 0
        self.training = split == "trainval"
        self.manifest = {}

        def read(part, filename):
            path = self.split_root / filename
            raw = path.read_bytes()
            samples, seen = [], set()
            for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    name, label_text = line.rsplit(maxsplit=1)
                    label = int(label_text)
                except ValueError as error:
                    raise ValueError(f"Invalid VTAB row: {path}:{number}") from error
                image = (self.root / name).resolve()
                if not image.is_relative_to(self.root):
                    raise ValueError(f"Image escapes dataset root: {path}:{number}")
                if image in seen:
                    raise ValueError(f"Duplicate VTAB image: {path}:{number}")
                if not 0 <= label < self.num_classes:
                    raise ValueError(f"Label out of range for {dataset}: {label}")
                seen.add(image)
                samples.append((image, label))
            if not samples:
                raise ValueError(f"Empty split: {path}")
            self.manifest[part] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                       count=len(samples))
            return samples, seen

        train, train_paths = read("trainval", "train800val200.txt")
        test, test_paths = read("test", "test.txt")
        if len(train) != 1000:
            raise ValueError(f"VTAB-1k requires 1000 training samples, got {len(train)}: {self.root}")
        if train_paths & test_paths:
            raise ValueError("VTAB test overlaps train800val200")
        # Some 1000-image training sets omit classes. Never renumber labels or
        # infer the output-head size from the subset of observed training labels.
        self.samples = train if self.training else test
