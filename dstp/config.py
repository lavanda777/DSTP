import argparse
import json
from pathlib import Path

from .seeds import load_seeds
from .benchmarks import VTAB_CLASSES


ROOT = Path(__file__).resolve().parents[1]

# Common Mixup defaults.
MIXUP_DEFAULTS = dict(mixup_prob=1.0, mixup_switch_prob=0.5, mixup_mode="batch")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DSTP project: ViT-B/16 FGVC / VTAB-1k training baseline")
    p.add_argument("--config", action="append", default=[], help="JSON override; may repeat, later wins")
    p.add_argument("--dataset", choices=["cub", "nabirds", "flowers", "dogs", "cars"] + list(VTAB_CLASSES), required=True)
    p.add_argument("--data-root", default="/home/pc/data/PEFT-method/GPS-main/datasets/FGVC")
    p.add_argument("--split-root", default=None, help="Optional split-metadata root; images still come from data-root")
    p.add_argument("--pretrained", default="/home/pc/data/PEFT-method/pretrained-model/VFM/ViT/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz")
    p.add_argument("--output", default=None, help="New run directory; existing nonempty directories require --resume")
    p.add_argument("--resume", default=None, help="Trusted DSTP baseline checkpoint, normally RUN/last.pt")
    p.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate, no batch-size scaling")
    p.add_argument("--head-lr", type=float, default=None,
                   help="Optional independent classifier learning rate")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW decoupled weight decay")
    p.add_argument("--epochs", type=int, default=30, help="Cosine schedule length, INCLUDING warmup")
    p.add_argument("--cooldown-epochs", type=int, default=0, help="Additional epochs at min_lr; total = epochs + cooldown")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--warmup-lr", type=float, default=1e-7)
    p.add_argument("--min-lr", type=float, default=1e-8)
    p.add_argument("--lr-interval", choices=["epoch", "step"], default="epoch")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--validation-batch-size", type=int, default=None,
                   help="Evaluation batch size; None uses the training batch size")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=load_seeds()[0],
                   help="Single run seed; default is first entry of configs/shared_seeds.json")
    p.add_argument("--amp", choices=["none", "float16", "bfloat16"], default="float16")
    p.add_argument("--device", default="cuda", help="Logical device AFTER CUDA_VISIBLE_DEVICES")
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--augmentation", choices=["simple", "vtab"], default="simple",
                   help="simple = random crop/flip; vtab = fixed 224x224 resize")
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--mixup-prob", type=float, default=MIXUP_DEFAULTS["mixup_prob"],
                   help="Probability of applying Mixup/CutMix; alpha and smoothing are separate")
    p.add_argument("--mixup-switch-prob", type=float, default=MIXUP_DEFAULTS["mixup_switch_prob"],
                   help="Probability of choosing CutMix when both mixing methods are enabled")
    p.add_argument("--mixup-mode", choices=["batch", "pair", "elem"],
                   default=MIXUP_DEFAULTS["mixup_mode"])
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.9998, help="0 disables EMA")
    p.add_argument("--checkpoint-every", type=int, default=0, help="Extra mid-epoch save interval in batches; 0 = epoch boundaries")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--stop-after-steps", type=int, default=0, help="Graceful test/interruption; full LR schedule is unchanged")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print-config", action="store_true", help="Resolve and validate config, then exit without training")
    preliminary, _ = p.parse_known_args(argv)
    overrides = {}
    suite = "vtab" if preliminary.dataset in VTAB_CLASSES else "fgvc"
    paths = [ROOT / "configs" / suite / f"{preliminary.dataset}.json"] + [Path(x) for x in preliminary.config]
    accepted = {a.dest: a for a in p._actions if a.dest not in {"help", "config", "print_config"}}
    for path in paths:
        values = json.loads(path.read_text())
        unknown = set(values) - set(accepted)
        if unknown:
            p.error(f"Unknown/unused keys in {path}: {sorted(unknown)}")
        overrides.update(values)
    p.set_defaults(**overrides)
    args = p.parse_args(argv)
    if args.dataset != preliminary.dataset:
        p.error("Configuration cannot change the dataset selected on the command line")
    for key, action in accepted.items():
        value = getattr(args, key)
        if action.choices and value not in action.choices:
            p.error(f"Invalid {key}: {value}")
    if args.epochs <= args.warmup_epochs or args.warmup_epochs < 0 or args.cooldown_epochs < 0:
        p.error("epochs > warmup_epochs >= 0 and cooldown_epochs >= 0 required")
    if min(args.lr, args.eps) <= 0 or min(args.min_lr, args.warmup_lr, args.weight_decay) < 0:
        p.error("Invalid LR, epsilon or weight decay")
    if max(args.min_lr, args.warmup_lr) > args.lr:
        p.error("min_lr and warmup_lr must not exceed the base LR")
    if args.head_lr is not None and not (args.head_lr > 0 and
            max(args.min_lr, args.warmup_lr) <= args.head_lr < float("inf")):
        p.error("head_lr must be finite, positive and >= min_lr/warmup_lr")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        p.error("betas must be in [0,1)")
    if not (0 <= args.ema_decay < 1 and 0 <= args.smoothing < 1 and 0 <= args.drop_path < 1):
        p.error("EMA, smoothing and drop_path must be in [0,1)")
    if not (0 <= args.mixup_prob <= 1 and 0 <= args.mixup_switch_prob <= 1):
        p.error("mixup_prob and mixup_switch_prob must be in [0,1]")
    if args.batch_size < 1 or args.workers < 0 or min(args.mixup, args.cutmix) < 0:
        p.error("Invalid batch/workers/mixing values")
    if args.validation_batch_size is not None and args.validation_batch_size < 1:
        p.error("validation_batch_size must be positive")
    if not 0 <= args.seed < 2 ** 32:
        p.error("seed must be in [0, 2**32)")
    if (args.mixup or args.cutmix) and args.batch_size % 2:
        p.error("timm batch Mixup/CutMix requires an even training batch size")
    if min(args.checkpoint_every, args.stop_after_steps) < 0 or args.log_every < 1:
        p.error("Invalid logging/checkpoint/stop interval")
    if args.amp != "none" and not args.device.startswith("cuda"):
        p.error("This runner supports AMP on CUDA only; use --amp none on CPU")
    if args.split_root is not None:
        path = Path(args.split_root).expanduser()
        args.split_root = str((ROOT / path).resolve())
    args.total_epochs = args.epochs + args.cooldown_epochs
    # One fixed GPS FGVC data/evaluation workflow, not tunable CLI options.
    # Persist these fields so resumes cannot silently change the experiment.
    args.train_split = "trainval"
    args.eval_split = "test"
    args.eval_interval = "epoch"
    args.eval_metric = "top1"
    return args
