# data/prepare_kancharla.py
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg, TASKS
from data.clean import clean_text

FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
OUT_NAME = {"train": "train", "validation": "val", "test": "test"}


def load_clean(split):
    df = pd.read_csv(cfg.paths.raw_dir / FILES[split])
    # the CSVs contain literal "\n" sequences; turn them into spaces before cleaning
    df["text"] = df["tweet"].astype(str).str.replace("\\n", " ", regex=False).map(clean_text)
    return df


def main():
    data = {s: load_clean(s) for s in FILES}

    # --- quick audit (we keep the authors' split unchanged so results stay comparable) ---
    for s, df in data.items():
        print(f"[audit] {s}: rows={len(df)} empty_after_cleaning={(df['text'] == '').sum()} "
              f"duplicate_texts={len(df) - df['text'].nunique()}")
    tr_texts, te_texts = set(data["train"]["text"]), set(data["test"]["text"])
    print(f"[audit] unique test texts also in train: {len(te_texts & tr_texts)} of {len(te_texts)}")

    # --- one folder per task: text,label,codemixed ---
    for task, info in TASKS.items():
        out_dir = cfg.paths.processed_dir / task
        out_dir.mkdir(parents=True, exist_ok=True)
        classes = info["classes"]
        for s, df in data.items():
            out = pd.DataFrame({
                "text": df["text"],
                "label": df[info["column"]].astype(int).map(lambda i: classes[i]),
                "codemixed": df["codemixed"],
            })
            out.to_csv(out_dir / f"{OUT_NAME[s]}.csv", index=False)
            print(f"[{task}] {OUT_NAME[s]}: {len(out)} rows  {out['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()