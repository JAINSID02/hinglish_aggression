# config.py
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Label ids follow the released CSVs (class counts match Table 1 of the paper)
TASKS = {
    "aggression": {"column": "aggression", "classes": ["NAG", "CAG", "OAG"]},  # 0, 1, 2
    "offense": {"column": "offense", "classes": ["NOT", "OFF"]},               # 0, 1
}

SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
PAD_ID = 2


def task_classes(task: str):
    return TASKS[task]["classes"]


@dataclass
class Paths:
    raw_dir: Path = ROOT / "data" / "raw" / "kancharla"      # train.csv, validation.csv, test.csv
    processed_dir: Path = ROOT / "data" / "processed"        # <task>/train.csv, val.csv, test.csv
    tokenizer_dir: Path = ROOT / "tokenizer" / "hinglish_bpe"
    pretrained_ckpt: Path = ROOT / "pretrained" / "model_best.pt"
    runs_dir: Path = ROOT / "runs"


@dataclass
class ModelConfig:
    # must match the pretrained backbone from the meme project
    vocab_size: int = 8000
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.3      # starting point; override with --dropout and judge on validation
    rope_base: float = 10000.0


@dataclass
class ClassifierConfig:
    max_len: int = 128
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 0.05
    epochs: int = 10
    patience: int = 3         # early stopping on validation macro-F1
    use_class_weights: bool = True
    seed: int = 42


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    cls: ClassifierConfig = field(default_factory=ClassifierConfig)


cfg = Config()