# eval/baseline.py
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg, TASKS, task_classes


def load(task, split):
    df = pd.read_csv(Path(cfg.paths.processed_dir) / task / f"{split}.csv", keep_default_na=False)
    classes = task_classes(task)
    return df["text"].astype(str).tolist(), [classes.index(l) for l in df["label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    task = ap.parse_args().task
    classes = task_classes(task)

    Xtr, ytr = load(task, "train")
    Xva, yva = load(task, "val")
    Xte, yte = load(task, "test")

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    Xtr_m = hstack([word.fit_transform(Xtr), char.fit_transform(Xtr)]).tocsr()   # fit on train only
    tf = lambda t: hstack([word.transform(t), char.transform(t)]).tocsr()
    Xva_m, Xte_m = tf(Xva), tf(Xte)

    best = None
    for C in [0.3, 1, 3, 10, 30]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr_m, ytr)
        f1 = f1_score(yva, clf.predict(Xva_m), average="macro", zero_division=0)
        print(f"C={C:<5} val macro-F1 = {f1:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    val_f1, C, clf = best

    pred = clf.predict(Xte_m)
    test_f1 = f1_score(yte, pred, average="macro", zero_division=0)
    test_acc = accuracy_score(yte, pred)
    maj = max(set(ytr), key=ytr.count)
    maj_pred = [maj] * len(yte)
    maj_f1 = f1_score(yte, maj_pred, average="macro", zero_division=0)
    maj_acc = accuracy_score(yte, maj_pred)

    print(f"\n[baseline:{task}] best C={C} (val macro-F1 {val_f1:.3f}). TEST:")
    print(f"  TF-IDF + LR : accuracy {test_acc:.3f}  macro-F1 {test_f1:.3f}")
    print(f"  majority    : accuracy {maj_acc:.3f}  macro-F1 {maj_f1:.3f}")
    print(classification_report(yte, pred, target_names=classes, zero_division=0, digits=3))

    out = Path(cfg.paths.runs_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"baseline_{task}.json").write_text(json.dumps({
        "task": task, "C": C, "val_macro_f1": val_f1, "test_macro_f1": test_f1, "test_acc": test_acc,
        "majority_macro_f1": maj_f1, "majority_acc": maj_acc}, indent=2))


if __name__ == "__main__":
    main()