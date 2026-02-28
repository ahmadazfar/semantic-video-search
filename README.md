# Semantic Video Search

A real-time video object detection, tracking, and natural-language search system. Upload any video, automatically detect and track every object with **RF-DETR** + **ByteTrack**, embed them with **CLIP**, and retrieve segments by typing plain-English queries — all through a Streamlit web UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

| Capability | Details |
|---|---|
| **Object Detection** | RF-DETR Medium (COCO-80 classes) running every *N* frames for efficiency |
| **Multi-Object Tracking** | ByteTrack with configurable lost-track buffer and activation threshold |
| **Optical Flow Interpolation** | Lucas-Kanade optical flow fills in detections between full RF-DETR frames, reducing GPU cost |
| **CLIP Embedding** | Each tracked object is cropped, padded to 224×224, and embedded with CLIP ViT-B/16 |
| **Vector Search** | ChromaDB (cosine similarity) stores per-object embeddings with first/last seen timestamps |
| **Natural Language Search** | Type a query like *"white car"* — CLIP encodes the text and retrieves the closest object segments |
| **Stationary Filtering** | Objects that haven't moved for a configurable number of checks are auto-indexed and skipped |
| **Queried Re-Detection** | After search, the system re-runs detection on the matched time window and highlights the target |
| **Streamlit Web UI** | Upload videos, index them, search across multiple indexed videos, and view annotated results |

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌───────────────┐
│  Upload &    │     │  RF-DETR     │     │  ByteTrack    │
│  Streamlit   │────▶│  Detection   │────▶│  Tracking     │
│  UI          │     │  (every Nth) │     │  + Optical    │
└──────────────┘     └──────────────┘     │  Flow interp  │
                                          └───────┬───────┘
                                                  │
                     ┌──────────────┐     ┌───────▼───────┐
                     │  ChromaDB    │◀────│  CLIP ViT-B/16│
                     │  Vector DB   │     │  Embedding    │
                     └──────┬───────┘     └───────────────┘
                            │
                     ┌──────▼───────┐
                     │  NL Query    │
                     │  Search &    │
                     │  Re-detect   │
                     └──────────────┘
```

---

## Project Structure

```
video-search-ai/
├── src/
│   ├── config.py              # Centralized configuration (all constants)
│   ├── model.py               # Lazy-loaded model singletons (CLIP, RF-DETR, VideoLLaVA)
│   ├── db.py                  # ChromaDB client & collection singletons
│   ├── state.py               # TrackState dataclass (per-video tracking buffers)
│   ├── annotation.py          # Main indexing pipeline (detect → track → crop → embed)
│   ├── tracking.py            # Object cropping, stationary filter, optical flow
│   ├── embedding.py           # CLIP embedding, ChromaDB storage & search
│   ├── queried_detection.py   # Re-detection on matched timestamp segments
│   ├── utils.py               # Helpers (timestamps, padded bbox, resize with padding)
│   └── streamlit/
│       ├── app.py             # Streamlit multi-page application
│       ├── landing.py         # Landing / home page
│       └── video_manager.py   # Video upload, registry I/O
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (NVIDIA, with CUDA toolkit installed)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/video-search-ai.git
cd video-search-ai

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run src/streamlit/app.py
```

The app will open at **http://localhost:8501**.

---

## Usage

1. **Upload & Index** — Upload a video (MP4/AVI), give it a name, and click *Start Backend Indexing*. The system runs RF-DETR + ByteTrack, crops every detected object, builds CLIP embeddings, and stores them in ChromaDB.

2. **Search Indexed Videos** — Select one or more indexed videos, type a natural-language query (e.g., *"red truck"*, *"person with backpack"*), and click *Run Search & Detection*. The system finds the best-matching object segment and re-runs detection to produce an annotated clip.

3. **Object Detection** — View the full annotated video with bounding boxes, class labels, confidence scores, and track IDs for every detected object.

---

## Configuration

All tunable parameters live in [src/config.py](src/config.py):

| Parameter | Default | Description |
|---|---|---|
| `DETECTION_INTERVAL` | 3 | Run RF-DETR every *N* frames (optical flow fills the gaps) |
| `DETECTION_THRESHOLD` | 0.3 | Minimum confidence for RF-DETR detections |
| `LOST_TRACK_BUFFER` | 180 | ByteTrack frames before a track is dropped |
| `LOST_FRAME_THRESHOLD` | 60 | Frames a track must be missing before its index is finalized |
| `CROP_TARGET_SIZE` | 224 | Crop resize target (CLIP input) |
| `CROP_PADDING_PERCENT` | 0.2 | Bounding-box padding before cropping |
| `STATIONARY_THRESHOLD_PX` | 3 | Pixel movement threshold for stationary filter |
| `SEARCH_N_RESULTS` | 3 | Top-*N* results returned from ChromaDB |

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | [RF-DETR Medium](https://github.com/roboflow/rf-detr) |
| Multi-Object Tracking | [ByteTrack](https://github.com/roboflow/supervision) (via Supervision) |
| Motion Interpolation | Lucas-Kanade Optical Flow (OpenCV) |
| Embeddings | [CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16) |
| Vector Database | [ChromaDB](https://www.trychroma.com/) |
| Multimodal (experimental) | [Video-LLaVA 7B](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf) |
| Web UI | [Streamlit](https://streamlit.io/) |
| Deep Learning | PyTorch, Transformers (HuggingFace) |
| Computer Vision | OpenCV, Pillow, Supervision |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
"# semantic-video-search" 

