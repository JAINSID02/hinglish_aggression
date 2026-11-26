# eval/metrics.py
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tokenizer.train_tokenizer import HinglishTokenizer
from model.classifier import MemeClassifier
from training.cls_dataset import make_cls_loaders
from training.utils import get_device


def load_classifier(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    c = ck["config"]
    model = MemeClassifier(c["backbone"], c["n_classes"], c["pad_id"]).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, c["categories"], c["task"]


@torch.no_grad()
def predict(model, loader, device):
    ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(device)).argmax(-1).cpu().tolist()
        ys += y.tolist()
    return np.array(ys), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder, e.g. runs/offense_pretrained_s42")
    run = Path(ap.parse_args().run)
    device = get_device()

    model, classes, task = load_classifier(run / "model_best.pt", device)
    tok = HinglishTokenizer()
    _, _, test_loader, _ = make_cls_loaders(tok, task)
    y, p = predict(model, test_loader, device)

    acc = accuracy_score(y, p)
    macro = f1_score(y, p, average="macro", zero_division=0)
    print(f"\n[{run.name}] TEST accuracy {acc:.3f} | macro-F1 {macro:.3f}")
    print(classification_report(y, p, target_names=classes, zero_division=0, digits=3))

    cm = confusion_matrix(y, p, labels=list(range(len(classes))))
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(run.name, fontsize=9)
    fig.tight_layout(); fig.savefig(run / "confusion_matrix.png", dpi=150); plt.close(fig)

    per_class = f1_score(y, p, average=None, labels=list(range(len(classes))), zero_division=0).tolist()
    (run / "test_result.json").write_text(json.dumps({
        "run": run.name, "task": task, "test_accuracy": acc, "test_macro_f1": macro,
        "per_class_f1": dict(zip(classes, per_class))}, indent=2))
    print(f"[metrics] saved test_result.json and confusion_matrix.png in {run}")


if __name__ == "__main__":
    main()