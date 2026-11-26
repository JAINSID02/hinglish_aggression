# eval/compare.py
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg

# Table 2 of Kancharla et al. (2025), combined data, macro-F1 x 100
PAPER = {
    "offense": {"BERT-base": 75.63, "RoBERTa-base": 76.31, "XLM-R-base": 76.38,
                "Hing-RoBERTa": 78.61, "MTTL-Hing-RoBERTa": 82.03},
    "aggression": {"BERT-base": 57.95, "RoBERTa-base": 60.70, "XLM-R-base": 60.58,
                   "Hing-RoBERTa": 64.85, "MTTL-Hing-RoBERTa": 67.01},
}


def load(p):
    return json.loads(Path(p).read_text())


def main():
    runs = Path(cfg.paths.runs_dir)
    rows = {}
    for task in PAPER:
        b = load(runs / f"baseline_{task}.json")
        r = {
            "Majority class (ours)": b["majority_macro_f1"] * 100,
            "TF-IDF + LR (ours)": b["test_macro_f1"] * 100,
            "Transformer, random init (ours)": load(runs / f"{task}_scratch_s42" / "test_result.json")["test_macro_f1"] * 100,
            "Transformer, pretrained (ours)": load(runs / f"{task}_pretrained_s42" / "test_result.json")["test_macro_f1"] * 100,
        }
        r.update({f"{k} (paper)": v for k, v in PAPER[task].items()})
        rows[task] = r

    names = sorted(rows["offense"], key=lambda k: -rows["offense"][k])
    lines = ["| System | Offensive | Aggression |", "|---|---|---|"]
    for n in names:
        lines.append(f"| {n} | {rows['offense'][n]:.1f} | {rows['aggression'][n]:.1f} |")
    text = "\n".join(lines)
    text += ("\n\nTest macro-F1 x 100 on the 829-tweet test set of Kancharla et al. (2025). Our numbers are single "
             "runs (seed 42); the paper's are the combined-data results of its Table 2.\n")
    (runs / "comparison.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()