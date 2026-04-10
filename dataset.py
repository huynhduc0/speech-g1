"""
dataset.py
──────────
NumbersDataset  – loads CSV + audio, extracts log-mel, tokenises labels.
collate_fn      – pads + stacks a batch.
spec_augment    – SpecAugment applied per-batch.
"""

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

import config as cfg
from text_utils import Tokenizer, number_to_text


# ── Mel feature extractor (shared, stateless) ─────────────────────────────────
_mel_transform = T.MelSpectrogram(
    sample_rate = cfg.SAMPLE_RATE,
    n_fft       = cfg.N_FFT,
    win_length  = cfg.WIN_LENGTH,
    hop_length  = cfg.HOP_LENGTH,
    n_mels      = cfg.N_MELS,
    f_min       = cfg.F_MIN,
    f_max       = cfg.F_MAX,
    power       = 2.0,
)


def _load_audio(path: str, target_sr: int = cfg.SAMPLE_RATE) -> torch.Tensor:
    """Load mono audio, resample to target_sr if needed. Returns (1, T)."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform   # (1, T)


def _wav_to_mel(waveform: torch.Tensor) -> torch.Tensor:
    """(1, T) → (T', n_mels) log-mel spectrogram."""
    mel = _mel_transform(waveform)          # (1, n_mels, T')
    mel = mel.squeeze(0).transpose(0, 1)   # (T', n_mels)
    mel = torch.log(mel.clamp(min=1e-9))
    return mel


# ── Dataset ───────────────────────────────────────────────────────────────────

class NumbersDataset(Dataset):
    """
    CSV columns expected: filename, transcription, [spk_id, gender, ext, samplerate]

    `transcription` is an integer (e.g. 280520).
    """

    def __init__(
        self,
        csv_path:    str,
        data_dir:    str,
        training:    bool = False,
        speed_perturb: bool = True,
    ):
        self.df       = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.training = training
        self.speed_p  = speed_perturb
        self.tok      = Tokenizer()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.data_dir, row["filename"])

        waveform = _load_audio(audio_path)

        # Speed perturbation (training only): ×{0.9, 1.0, 1.1}
        if self.training and self.speed_p and random.random() < 0.5:
            rate = random.choice([0.9, 1.1])
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq = int(cfg.SAMPLE_RATE * rate),
                new_freq  = cfg.SAMPLE_RATE,
            )

        mel    = _wav_to_mel(waveform)       # (T', 80)
        target = self.tok.encode(number_to_text(int(row["transcription"])))

        return mel, torch.tensor(target, dtype=torch.long)

    def get_spk_id(self, idx: int) -> str:
        return str(self.df.iloc[idx].get("spk_id", ""))


# ── Collation ─────────────────────────────────────────────────────────────────

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        mels        : (B, T_max, n_mels)  – zero-padded
        mel_lengths : (B,)                – valid frame counts
        targets     : (sum of target lens,)
        tgt_lengths : (B,)
    """
    mels, targets = zip(*batch)

    mel_lengths = torch.tensor([m.size(0) for m in mels], dtype=torch.long)
    tgt_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)

    # Pad mels
    max_t = mel_lengths.max().item()
    n_mel = mels[0].size(1)
    padded = torch.zeros(len(mels), max_t, n_mel)
    for i, m in enumerate(mels):
        padded[i, : m.size(0)] = m

    targets_cat = torch.cat(targets)

    return padded, mel_lengths, targets_cat, tgt_lengths


# ── SpecAugment (applied to batch after collation) ───────────────────────────

def spec_augment(
    mels:            torch.Tensor,   # (B, T, F)
    mel_lengths:     torch.Tensor,   # (B,)
    freq_mask_param: int = cfg.FREQ_MASK_PARAM,
    time_mask_param: int = cfg.TIME_MASK_PARAM,
    num_freq_masks:  int = cfg.NUM_FREQ_MASKS,
    num_time_masks:  int = cfg.NUM_TIME_MASKS,
) -> torch.Tensor:
    """In-place SpecAugment on a batch tensor."""
    B, T, F = mels.shape
    mels = mels.clone()
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, F - f)
        mels[:, :, f0 : f0 + f] = 0.0

    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(T - t, 0))
        mels[:, t0 : t0 + t, :] = 0.0

    return mels


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = NumbersDataset(cfg.TRAIN_CSV, cfg.DATA_DIR, training=True)
    print(f"Train samples: {len(ds)}")
    mel, tgt = ds[0]
    print(f"  mel={mel.shape}  target_len={tgt.shape[0]}")

    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    mels, ml, tgts, tl = next(iter(loader))
    mels = spec_augment(mels, ml)
    print(f"  batch mel={mels.shape}  mel_lens={ml}  tgt_lens={tl}")
