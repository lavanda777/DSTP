"""Shared seed configuration for training launchers."""

import json
from pathlib import Path


DEFAULT_SEED_FILE = Path(__file__).resolve().parents[1] / "configs/shared_seeds.json"


def validate_seeds(seeds):
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("A nonempty seed list is required")
    if any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds would repeat an experiment")
    return seeds


def load_seeds(path=DEFAULT_SEED_FILE):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or set(payload) != {"seeds"}:
        raise ValueError("Seed file must contain exactly a 'seeds' list")
    return validate_seeds(payload["seeds"])
