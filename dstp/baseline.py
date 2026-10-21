"""Plain AdamW baseline; replace this hook when the DSTP method is defined."""
import torch


def build_optimizer(model, args):
    # This is ordinary full fine-tuning, not an implementation of the DSTP method.
    model.requires_grad_(True)
    backbone, head = [], []
    manifest = {"backbone": [], "head": []}
    for name, parameter in model.named_parameters():
        key = "head" if name.startswith("head.") else "backbone"
        (head if key == "head" else backbone).append(parameter)
        manifest[key].append(dict(name=name, shape=list(parameter.shape), numel=parameter.numel()))
    groups = [dict(params=backbone, lr=args.lr, name="backbone"),
              dict(params=head, lr=args.head_lr if args.head_lr is not None else args.lr, name="head")]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2),
                                 eps=args.eps, weight_decay=args.weight_decay)
    manifest["total_parameters"] = sum(p.numel() for p in model.parameters())
    manifest["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    manifest["method"] = "adamw_full_finetune_baseline"
    return optimizer, manifest
