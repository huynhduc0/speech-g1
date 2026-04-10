import os

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/kaggle/input/asr-2026-spoken-numbers-recognition-challenge",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
DEV_CSV   = os.path.join(DATA_DIR, "dev.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

# ── Audio / feature extraction ───────────────────────────────────────────────
SAMPLE_RATE  = 16_000
N_MELS       = 80
N_FFT        = 512
HOP_LENGTH   = 160   # 10 ms
WIN_LENGTH   = 400   # 25 ms
F_MIN        = 80.0
F_MAX        = 7_600.0

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_DIM   = 144
NUM_HEADS   = 4
NUM_LAYERS  = 8
FF_DIM      = MODEL_DIM * 4   # 576
CONV_KERNEL = 31
DROPOUT     = 0.1

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 3e-4
WARMUP_STEPS  = 4_000
MAX_EPOCHS    = 60
GRAD_CLIP     = 5.0
WEIGHT_DECAY  = 1e-2
NUM_WORKERS   = 4

# ── SpecAugment ───────────────────────────────────────────────────────────────
FREQ_MASK_PARAM  = 15
TIME_MASK_PARAM  = 50
NUM_FREQ_MASKS   = 2
NUM_TIME_MASKS   = 2

# ── Checkpointing ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
