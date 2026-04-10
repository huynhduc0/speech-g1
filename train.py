"""
train.py
────────
Full training loop for ConformerCTC on the Russian spoken numbers dataset.

Usage:
    python train.py [--epochs N] [--batch-size B] [--lr LR]

Runs on GPU if available, falls back to CPU.
Saves best checkpoint (lowest dev CER) to OUTPUT_DIR/best_model.pt.
"""

from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import NumbersDataset, collate_fn, spec_augment
from model import ConformerCTC
from text_utils import (
    VOCAB_SIZE, BLANK_ID, Tokenizer,
    ctc_greedy_decode, text_to_number, number_to_text,
)


# ── LR schedule: linear warmup + cosine decay ────────────────────────────────

def _build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ── CER metric ───────────────────────────────────────────────────────────────

def _cer(ref: str, hyp: str) -> float:
    """Character Error Rate via edit distance."""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    r, h = list(ref), list(hyp)
    # Dynamic programming
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(r)][len(h)] / len(r)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:  ConformerCTC,
    loader: DataLoader,
    device: torch.device,
    tok:    Tokenizer,
    df_dev: pd.DataFrame,
) -> dict[str, float]:
    """Returns dict with 'cer_inD', 'cer_ooD', 'score' (harmonic mean × 100)."""
    model.eval()

    # Collect per-sample CER split by spk_id domain
    # We need to know which spk_ids were in the training set.
    # We approximate: load train spk_ids from train.csv
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    train_spk = set(train_df["spk_id"].unique())

    cer_inD: list[float] = []
    cer_ooD: list[float] = []

    sample_idx = 0
    for mels, mel_lengths, targets, tgt_lengths in loader:
        mels       = mels.to(device)
        mel_lengths = mel_lengths.to(device)

        log_probs, out_lens = model(mels, mel_lengths)

        # Greedy decode
        hyps = ctc_greedy_decode(log_probs.cpu(), out_lens.cpu())

        # Split targets back per sample
        tgt_offset = 0
        for b in range(mels.size(0)):
            tl   = tgt_lengths[b].item()
            ref_ids = targets[tgt_offset : tgt_offset + tl].tolist()
            tgt_offset += tl

            ref_text = tok.decode(ref_ids)
            hyp_text = tok.decode(hyps[b])
            cer_val  = _cer(ref_text, hyp_text)

            spk = df_dev.iloc[sample_idx].get("spk_id", "")
            if spk in train_spk:
                cer_inD.append(cer_val)
            else:
                cer_ooD.append(cer_val)
            sample_idx += 1

    avg_inD = sum(cer_inD) / len(cer_inD) if cer_inD else 0.0
    avg_ooD = sum(cer_ooD) / len(cer_ooD) if cer_ooD else 0.0
    denom   = avg_inD + avg_ooD
    score   = 2 * avg_inD * avg_ooD / denom * 100.0 if denom > 0 else 0.0

    return {"cer_inD": avg_inD, "cer_ooD": avg_ooD, "score": score}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # ── Datasets / loaders ────────────────────────────────────────────────────
    train_ds = NumbersDataset(cfg.TRAIN_CSV, cfg.DATA_DIR, training=True)
    dev_ds   = NumbersDataset(cfg.DEV_CSV,   cfg.DATA_DIR, training=False)
    dev_df   = pd.read_csv(cfg.DEV_CSV)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = cfg.NUM_WORKERS,
        collate_fn  = collate_fn,
        pin_memory  = device.type == "cuda",
        drop_last   = True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = cfg.NUM_WORKERS,
        collate_fn  = collate_fn,
        pin_memory  = device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConformerCTC(
        vocab_size  = VOCAB_SIZE,
        n_mels      = cfg.N_MELS,
        model_dim   = cfg.MODEL_DIM,
        num_heads   = cfg.NUM_HEADS,
        num_layers  = cfg.NUM_LAYERS,
        ff_dim      = cfg.FF_DIM,
        conv_kernel = cfg.CONV_KERNEL,
        dropout     = cfg.DROPOUT,
    ).to(device)

    total_params = model.count_params()
    print(f"Parameters: {total_params:,}  ({total_params / 1e6:.2f} M)")
    assert total_params <= 5_000_000, "Model exceeds 5 M parameter limit!"

    # ── Loss / optim ──────────────────────────────────────────────────────────
    ctc_loss  = nn.CTCLoss(blank=BLANK_ID, reduction="mean", zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY)

    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = min(cfg.WARMUP_STEPS, total_steps // 10)
    scheduler     = _build_scheduler(optimizer, warmup_steps, total_steps)
    scaler        = GradScaler(enabled=(device.type == "cuda"))

    # ── Training ──────────────────────────────────────────────────────────────
    best_score  = float("inf")   # lower CER = better; score is harmonic-mean CER
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss  = 0.0
        t0          = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for mels, mel_lengths, targets, tgt_lengths in pbar:
            mels        = spec_augment(mels, mel_lengths).to(device)
            mel_lengths = mel_lengths.to(device)
            targets     = targets.to(device)
            tgt_lengths = tgt_lengths.to(device)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                log_probs, out_lens = model(mels, mel_lengths)
                loss = ctc_loss(log_probs, targets, out_lens, tgt_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0

        # ── Evaluate ──────────────────────────────────────────────────────────
        metrics = evaluate(model, dev_loader, device, Tokenizer(), dev_df)
        score   = metrics["score"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"CER inD={metrics['cer_inD']*100:.2f}% "
            f"ooD={metrics['cer_ooD']*100:.2f}% "
            f"score={score:.2f}% | "
            f"{elapsed:.0f}s"
        )

        # Save checkpoint every epoch
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "score":       score,
        }, ckpt_path)

        # Save best model
        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            print(f"  → New best! score={best_score:.2f}%  saved to {cfg.BEST_MODEL_PATH}")

    print(f"\nTraining done. Best score: {best_score:.2f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=cfg.MAX_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=cfg.LEARNING_RATE)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
