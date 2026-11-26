# training/finetune_cls.py
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg, TASKS, task_classes, PAD_ID
from tokenizer.train_tokenizer import HinglishTokenizer
from model.classifier import MemeClassifier
from training.cls_dataset import make_cls_loaders
from training.utils import set_seed, get_device, WarmupCosine, build_optimizer, save_checkpoint


@torch.no_grad()
def evaluate(model, loader, device, weights=None):
    """Returns (loss, accuracy, macro_f1)."""
    model.eval()
    preds, trues, total, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += F.cross_entropy(logits, y, weight=weights, reduction="sum").item()
        n += y.numel()
        preds += logits.argmax(-1).tolist()
        trues += y.tolist()
    acc = sum(p == t for p, t in zip(preds, trues)) / max(1, n)
    return total / max(1, n), acc, f1_score(trues, preds, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--from_scratch", action="store_true", help="random init (no pretrained backbone)")
    ap.add_argument("--pretrained", default=str(cfg.paths.pretrained_ckpt))
    ap.add_argument("--seed", type=int, default=cfg.cls.seed)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dropout is not None: cfg.model.dropout = args.dropout
    if args.lr is not None: cfg.cls.lr = args.lr
    if args.epochs is not None: cfg.cls.epochs = args.epochs
    if args.weight_decay is not None: cfg.cls.weight_decay = args.weight_decay
    cc = cfg.cls

    init = "scratch" if args.from_scratch else "pretrained"
    out_dir = Path(args.out or cfg.paths.runs_dir / f"{args.task}_{init}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    classes = task_classes(args.task)

    tok = HinglishTokenizer()
    train_loader, val_loader, _, weights = make_cls_loaders(tok, args.task)
    if weights is not None:
        weights = weights.to(device)

    model = MemeClassifier.from_config(cfg.model, len(classes), PAD_ID).to(device)
    if args.from_scratch:
        print(f"[finetune:{args.task}] RANDOM init (no pretraining)")
    else:
        model.load_pretrained_backbone(args.pretrained, map_location=device)

    opt = build_optimizer(model, cc.lr, cc.weight_decay, betas=(0.9, 0.999))
    total_steps = cc.epochs * len(train_loader)
    sched = WarmupCosine(opt, cc.lr, warmup_steps=max(1, total_steps // 10), total_steps=total_steps)

    classifier_cfg = {"backbone": model.backbone_cfg, "n_classes": len(classes), "pad_id": PAD_ID,
                      "categories": classes, "task": args.task}
    best_f1, best_epoch, bad, step = -1.0, 0, 0, 0
    best_path = out_dir / "model_best.pt"

    for epoch in range(1, cc.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            sched.set(step)
            loss = F.cross_entropy(model(x), y, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            run_loss += loss.item() * y.numel()
            seen += y.numel()

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, weights)
        print(f"epoch {epoch:2d}/{cc.epochs} | train loss {run_loss / seen:.4f} | val loss {val_loss:.4f} "
              f"| val acc {val_acc:.3f} | val macro-F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            save_checkpoint(best_path, model, step=step, config=classifier_cfg,
                            extra={"val_macro_f1": val_f1, "epoch": epoch})
            print("           new best macro-F1 -> model_best.pt")
        else:
            bad += 1
            if bad >= cc.patience:
                print(f"[finetune:{args.task}] early stop (no gain for {cc.patience} epochs)")
                break

    (out_dir / "result.json").write_text(json.dumps({
        "task": args.task, "init": init, "seed": args.seed, "dropout": cfg.model.dropout,
        "lr": cc.lr, "weight_decay": cc.weight_decay, "best_val_macro_f1": best_f1, "best_epoch": best_epoch}, indent=2))
    print(f"[finetune:{args.task}] best val macro-F1 = {best_f1:.3f} at epoch {best_epoch} -> {out_dir}")


if __name__ == "__main__":
    main()