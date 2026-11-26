# training/cls_dataset.py
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg, task_classes
from tokenizer.train_tokenizer import HinglishTokenizer


class TextDataset(Dataset):
    """Reads <task>/<split>.csv (columns: text,label,codemixed). Each text is tokenized once,
    truncated to max_len-1 tokens, and ends with <eos> (same as pretraining)."""

    def __init__(self, csv_path, tok: HinglishTokenizer, max_len: int, label2id: dict):
        df = pd.read_csv(csv_path, keep_default_na=False)
        self.texts = df["text"].astype(str).tolist()
        self.labels = [label2id[l] for l in df["label"]]
        self.codemixed = df["codemixed"].tolist()
        self.items = [tok.encode(t)[: max_len - 1] + [tok.eos_id] for t in self.texts]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], self.labels[i]


def make_collate(pad_id: int):
    def collate(batch):
        """Right-pad to the longest sequence in the batch; the classifier ignores pads when pooling."""
        max_t = max(len(ids) for ids, _ in batch)
        x = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor([lab for _, lab in batch], dtype=torch.long)
        return x, y
    return collate


def compute_class_weights(labels, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalized to average 1 (same idea as Eq. 2 of Kancharla et al.)."""
    counts = torch.bincount(torch.tensor(labels), minlength=n_classes).float().clamp_min(1.0)
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()


def make_cls_loaders(tok: HinglishTokenizer, task: str):
    classes = task_classes(task)
    label2id = {c: i for i, c in enumerate(classes)}
    cc = cfg.cls
    collate = make_collate(tok.pad_id)
    d = Path(cfg.paths.processed_dir) / task

    train_ds = TextDataset(d / "train.csv", tok, cc.max_len, label2id)
    val_ds = TextDataset(d / "val.csv", tok, cc.max_len, label2id)
    test_ds = TextDataset(d / "test.csv", tok, cc.max_len, label2id)

    train_loader = DataLoader(train_ds, batch_size=cc.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cc.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cc.batch_size, shuffle=False, collate_fn=collate)

    weights = compute_class_weights(train_ds.labels, len(classes)) if cc.use_class_weights else None
    print(f"[data:{task}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={classes}")
    if weights is not None:
        print(f"[data:{task}] class weights: {[round(w, 2) for w in weights.tolist()]}")
    return train_loader, val_loader, test_loader, weights


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="aggression", choices=["aggression", "offense"])
    args = ap.parse_args()
    tok = HinglishTokenizer()
    tr, va, te, w = make_cls_loaders(tok, args.task)
    x, y = next(iter(tr))
    longest = max(len(ids) for ids in tr.dataset.items)
    print(x.shape, y.tolist()[:8], "| longest tweet (tokens):", longest, "| max_len:", cfg.cls.max_len)