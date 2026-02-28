"""
Centralized configuration for Video Search AI.

All paths, model names, and processing parameters live here
so nothing is hardcoded across the core modules.
"""

import torch

# ── Device ───────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model names ──────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
RFDETR_RESOLUTION = 640
VIDEO_LLAVA_MODEL_NAME = "LanguageBind/Video-LLaVA-7B-hf"

# ── ChromaDB ─────────────────────────────────────────────────────────────
CHROMA_DB_PATH = "./video_index_db"
CHROMA_COLLECTION_NAME = "video_moments"

# ── Data directories ─────────────────────────────────────────────────────
CLIPS_DIR = "./data/clips"
ANNOTATED_VIDEOS_DIR = "./data/annotated_videos"
CROPPED_OBJECTS_DIR = "./data/cropped_objects"
QUERIED_DETECTIONS_DIR = "./data/queried_detections"
UPLOADED_VIDEOS_DIR = "./data/uploaded_videos"

# ── Video registry ───────────────────────────────────────────────────────
VIDEO_REGISTRY_PATH = "./video_registry.json"

# ── Detection & Tracking ─────────────────────────────────────────────────
DETECTION_INTERVAL = 3              # Run RF-DETR every N frames
DETECTION_THRESHOLD = 0.3
LOST_TRACK_BUFFER = 180             # ByteTrack lost-track buffer (frames)
LOST_FRAME_THRESHOLD = 60           # Frames before a track is finalised

# ── Cropping & Embedding ─────────────────────────────────────────────────
CROP_TARGET_SIZE = 224               # Resize crops to this square
CROP_PADDING_PERCENT = 0.2           # Bounding-box padding
EMBEDDING_BATCH_SIZE = 32            # Batch size for CLIP image embedding
CROP_INTERVAL_DIVISOR = 2          # Crop every N frames (N = fps / divisor)

# ── Stationary filter ────────────────────────────────────────────────────
STATIONARY_THRESHOLD_PX = 3        # Pixel distance to be considered "moving"
STATIONARY_CHECK_COUNT = 10         # Consecutive checks before marked stationary

# ── Video clipping ───────────────────────────────────────────────────────
MAX_CLIP_DURATION_SEC = 60           # Max seconds when clipping raw video

# ── Search ───────────────────────────────────────────────────────────────
SEARCH_N_RESULTS = 3                 # Top-N results from ChromaDB

# ── Annotation styling ───────────────────────────────────────────────────
COLOR_PALETTE_HEX = [
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
]
LABEL_TEXT_SCALE = 0.4
LABEL_TEXT_THICKNESS = 1
LABEL_TEXT_PADDING = 4

# ── Video-LLaVA ─────────────────────────────────────────────────────────
LLAVA_NUM_FRAMES = 8                 # Frames sampled for Video-LLaVA input
LLAVA_MAX_NEW_TOKENS = 100           # Max tokens for generation
