from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# Image params
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32
NUM_CLASSES = 3  # grade_a, grade_b, grade_c

# Training
EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 7

# Augmentation
AUGMENTATION_PARAMS = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "fill_mode": "nearest"
}

# Model
MODEL_DIR = BASE_DIR / "models"
MODEL_NAME = "cnn_crumb_rubber.h5"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

# Random seed
SEED = 42
