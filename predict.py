"""
predict.py
──────────
Load best checkpoint → run inference on test set → generate submission.csv.

Usage:
    python predict.py [--model-path PATH] [--batch-size N]

Output:
    OUTPUT_DIR/submission.csv
"""

from __future__ import annotations
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config as cfg
from dataset import _load_audio, _wav_to_mel
from model import ConformerCTC
from text_utils import (
    VOCAB_SIZE, BLANK_ID, Tokenizer,
    ctc_greedy_decode, text_to_number,
)


# ── Inference-only dataset (no labels needed) ─────────────────────────────────

class TestDataset(Dataset):
    def __init__(self, csv_path: str, data_dir: str):
        self.df       = pd.read_csv(csv_path)
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row        = self.df.iloc[idx]
        audio_path = os.path.join(self.data_dir, row["filename"])
        waveform   = _load_audio(audio_path)
        mel        = _wav_to_mel(waveform)   # (T', 80)
        return mel, row["filename"]


def _test_collate(batch):
    mels, filenames = zip(*batch)
    lengths = torch.tensor([m.size(0) for m in mels], dtype=torch.long)
    max_t   = lengths.max().item()
    padded  = torch.zeros(len(mels), max_t, mels[0].size(1))
    for i, m in enumerate(mels):
        padded[i, : m.size(0)] = m
    return padded, lengths, list(filenames)


# ── Post-processing: decoded text → integer ───────────────────────────────────

def _safe_int(text: str, filename: str) -> int:
    """Convert decoded Russian text to integer; fall back to 1000 on failure."""
    val = text_to_number(text)
    if val <= 0 or val > 999_999:
        print(f"  [warn] bad parse for {filename!r}: {text!r} → fallback 1000")
        return 1_000
    return val


# ── Main ──────────────────────────────────────────────────────────────────────

def predict(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = ConformerCTC(
        vocab_size  = VOCAB_SIZE,
        n_mels      = cfg.N_MELS,
        model_dim   = cfg.MODEL_DIM,
        num_heads   = cfg.NUM_HEADS,
        num_layers  = cfg.NUM_LAYERS,
        ff_dim      = cfg.FF_DIM,
        conv_kernel = cfg.CONV_KERNEL,
        dropout     = 0.0,          # no dropout at inference
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.model_path}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    test_ds = TestDataset(cfg.TEST_CSV, cfg.DATA_DIR)
    loader  = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = cfg.NUM_WORKERS,
        collate_fn  = _test_collate,
        pin_memory  = device.type == "cuda",
    )

    tok = Tokenizer()
    rows: list[dict] = []

    with torch.no_grad():
        for mels, lengths, filenames in tqdm(loader, desc="Inference"):
            mels    = mels.to(device)
            lengths = lengths.to(device)

            log_probs, out_lens = model(mels, lengths)
            hyps = ctc_greedy_decode(log_probs.cpu(), out_lens.cpu())

            for filename, ids in zip(filenames, hyps):
                text = tok.decode(ids)
                pred = _safe_int(text, filename)
                rows.append({"filename": filename, "transcription": pred})

    # ── Save submission ───────────────────────────────────────────────────────
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    sub_path = os.path.join(cfg.OUTPUT_DIR, "submission.csv")
    pd.DataFrame(rows).to_csv(sub_path, index=False)
    print(f"Saved {len(rows)} predictions → {sub_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default=cfg.BEST_MODEL_PATH,
        help="Path to model state dict or full checkpoint",
    )
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    predict(parse_args())
