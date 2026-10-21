"""Sequential, shell-free multi-seed launcher.

Use --dry-run to print a JSON command plan without running anything. A literal
{seed} in arguments is replaced for each run; --seed is appended explicitly.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

from .seeds import DEFAULT_SEED_FILE, load_seeds, validate_seeds


def seed_commands(command, seeds, environ):
    validate_seeds(seeds)
    if not command:
        raise ValueError("Specify a command after --")
    if environ.get("RESUME") or any(t == "--resume" or t.startswith("--resume=") for t in command):
        raise ValueError("Resume one checkpoint with its own seed; multi-seed resume is not allowed")
    if any(t == "--seed" or t.startswith("--seed=") for t in command):
        raise ValueError("Use --seeds before --, not --seed inside the wrapped command")
    if environ.get("RUN_DIR") and "{seed}" not in environ["RUN_DIR"]:
        raise ValueError("RUN_DIR must contain {seed} so runs have independent outputs")
    for i, token in enumerate(command):
        if token == "--output":
            if i + 1 >= len(command) or "{seed}" not in command[i + 1]:
                raise ValueError("An explicit --output must contain {seed}")
        elif token.startswith("--output=") and "{seed}" not in token:
            raise ValueError("An explicit --output must contain {seed}")
    return [{"seed": seed, "argv": [t.replace("{seed}", str(seed)) for t in command] +
             ["--seed", str(seed)]} for seed in seeds]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--seeds", nargs="+", type=int, help="Override shared file for this sweep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        seeds = validate_seeds(args.seeds) if args.seeds is not None else load_seeds(args.seed_file)
        plan = seed_commands(command, seeds, os.environ)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(dict(seed_file=str(args.seed_file) if args.seeds is None else None,
                          seeds=seeds, dry_run=args.dry_run, commands=plan), indent=2), flush=True)
    if not args.dry_run:
        for job in plan:
            env = dict(os.environ, SEED=str(job["seed"]))
            if env.get("RUN_DIR"):
                env["RUN_DIR"] = env["RUN_DIR"].replace("{seed}", str(job["seed"]))
            subprocess.run(job["argv"], check=True, env=env)


if __name__ == "__main__":
    main()
