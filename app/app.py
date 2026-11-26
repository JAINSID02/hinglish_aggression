# app/app.py
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg
from data.clean import clean_text
from eval.metrics import load_classifier
from tokenizer.train_tokenizer import HinglishTokenizer
from training.utils import get_device

st.set_page_config(page_title="Hinglish Tweet Analyzer", layout="wide")

LABELS = {"NOT": "Not offensive", "OFF": "Offensive",
          "NAG": "Not aggressive", "CAG": "Covertly aggressive", "OAG": "Overtly aggressive"}
COLORS = {"NOT": "#2e7d5b", "NAG": "#2e7d5b", "CAG": "#c78a1a", "OAG": "#b3392f", "OFF": "#b3392f"}
TASK_TITLES = {"offense": "Offensive language", "aggression": "Aggression level"}
LOW_CONF = 0.60

EXAMPLES = {
    "Select an example": "",
    "Neutral comment": "great speech by the minister today, hope the policy actually helps the farmers",
    "Casual Hinglish": "kal match dekhte hain sab milke, snacks main laa raha hoon",
    "Insulting tone": "tum log kuch nahi kar sakte, sab ke sab bekar ho aur bekar hi rahoge",
    "Sarcastic remark": "wah kya kaam kiya hai, itna bada nuksaan karke bhi sar utha ke ghoom rahe ho",
}

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1100px;}
.app-title {font-size: 1.9rem; font-weight: 650; margin-bottom: 0.1rem;}
.app-sub {color: #5b6675; margin-bottom: 1.2rem;}
.card {border: 1px solid #dde3ea; border-radius: 10px; padding: 18px 20px; background: #ffffff;}
.card h4 {margin: 0 0 4px 0; font-size: 0.8rem; letter-spacing: 0.07em; text-transform: uppercase; color: #5b6675;}
.verdict {font-size: 1.5rem; font-weight: 650; margin: 2px 0 12px 0;}
.row {display: flex; align-items: center; gap: 10px; margin: 7px 0; font-size: 0.9rem; color: #1c2530;}
.lab {width: 150px;}
.track {flex: 1; background: #edf1f5; border-radius: 6px; height: 10px; overflow: hidden;}
.fill {height: 100%; border-radius: 6px;}
.pct {width: 56px; text-align: right; font-variant-numeric: tabular-nums;}
.note {color: #8a5a00; font-size: 0.82rem; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models...")
def load_tokenizer():
    return HinglishTokenizer()


@st.cache_resource(show_spinner="Loading models...")
def load_models(init: str):
    device = get_device()
    models = {}
    for task in ("offense", "aggression"):
        ckpt = Path(cfg.paths.runs_dir) / f"{task}_{init}_s42" / "model_best.pt"
        if ckpt.exists():
            model, classes, _ = load_classifier(ckpt, device)
            models[task] = (model, classes)
    return models, device


def predict(text, models, tok, device):
    ids = tok.encode(clean_text(text))[: cfg.cls.max_len - 1] + [tok.eos_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = {}
    with torch.no_grad():
        for task, (model, classes) in models.items():
            p = F.softmax(model(x), dim=-1)[0].cpu().tolist()
            out[task] = dict(zip(classes, p))
    return out


def card_html(task, probs):
    top = max(probs, key=probs.get)
    rows = "".join(
        f'<div class="row"><span class="lab">{LABELS[k]}</span>'
        f'<div class="track"><div class="fill" style="width:{v * 100:.1f}%;background:{COLORS[k]}"></div></div>'
        f'<span class="pct">{v * 100:.1f}%</span></div>' for k, v in probs.items())
    note = '<div class="note">Low confidence: treat this prediction with caution.</div>' if probs[top] < LOW_CONF else ""
    return (f'<div class="card"><h4>{TASK_TITLES[task]}</h4>'
            f'<div class="verdict" style="color:{COLORS[top]}">{LABELS[top]}</div>{rows}{note}</div>')


def load_example():
    ex = EXAMPLES[st.session_state.example]
    if ex:
        st.session_state.text = ex


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### Model")
    version = st.radio("Version", ["Pretrained backbone", "Random initialization"], label_visibility="collapsed")
    init = "pretrained" if version.startswith("Pretrained") else "scratch"
    st.caption("5.26M-parameter transformer built from scratch (rotary embeddings, RMSNorm, SwiGLU), "
               "trained on a laptop CPU. Input is cut at 128 tokens.")
    st.markdown("---")
    st.caption("Research prototype. Predictions can be wrong and must not be used to moderate content "
               "or make decisions about people.")

# ---------------- header ----------------
st.markdown('<div class="app-title">Hinglish Tweet Analyzer</div>'
            '<div class="app-sub">Offensive-language and aggression detection for Hindi-English code-mixed text</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Analyze", "Model results", "About"])

with tab1:
    st.selectbox("Example tweets", list(EXAMPLES), key="example", on_change=load_example)
    text = st.text_area("Tweet text", key="text", height=120, placeholder="Type or paste a Hinglish or English tweet")
    run = st.button("Analyze", type="primary", disabled=not text.strip())

    if run:
        models, device = load_models(init)
        if not models:
            st.error("No trained checkpoints found in the runs folder. Train the models first "
                     "(see the README), then reload this page.")
        else:
            probs = predict(text, models, load_tokenizer(), device)
            cols = st.columns(len(probs), gap="large")
            for col, (task, p) in zip(cols, probs.items()):
                col.markdown(card_html(task, p), unsafe_allow_html=True)
            st.caption("Percentages are the model's output probabilities. They are not calibrated "
                       "and small models are often over-confident.")
            with st.expander("Text as seen by the model"):
                st.code(clean_text(text) or "(empty after cleaning)", language=None)

with tab2:
    st.markdown("#### Test macro-F1 on the 829-tweet benchmark split of Kancharla et al. (2025)")
    df = pd.DataFrame([
        ("MTTL-Hing-RoBERTa", "Paper", 82.0, 67.0), ("Hing-RoBERTa", "Paper", 78.6, 64.8),
        ("XLM-R-base", "Paper", 76.4, 60.6), ("RoBERTa-base", "Paper", 76.3, 60.7),
        ("BERT-base", "Paper", 75.6, 58.0), ("TF-IDF + logistic regression", "Ours", 72.8, 59.0),
        ("Transformer, random init", "Ours", 70.9, 50.8), ("Transformer, pretrained", "Ours", 70.2, 51.6),
        ("Majority class", "Ours", 41.2, 21.0)],
        columns=["System", "Source", "Offensive", "Aggression"])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.markdown(
        "- A simple TF-IDF baseline beats our small transformer on both tasks, and on aggression it also beats BERT-base.\n"
        "- Pretraining on 1.7M tokens makes no consistent difference (under 1 point).\n"
        "- The paper's pretrained models lead our transformer by about 12 points (offensive) and 15 points (aggression).")
    st.caption("Our results are single runs (seed 42). The paper's numbers are the combined-data results of its Table 2.")

with tab3:
    st.markdown("""
**Tasks.** Offensive language (2 classes) and aggression (3 classes: not, covertly and overtly aggressive)
on Hindi-English code-mixed tweets.

**Model.** A 5.26M-parameter transformer written from scratch in PyTorch, pretrained on about 1.7M tokens of
Hinglish text and fine-tuned for each task. It runs on a CPU.

**Data.** The benchmark of Nafis et al. (2023), using the text splits released by Kancharla et al. (2025):
6,624 training, 828 validation and 829 test tweets. The data is not distributed with this app.

**Limitations.** Single training run; no confidence intervals; a small model compared with the pretrained models
in the paper; tweets longer than 128 tokens are truncated; the data contains offensive language.
""")