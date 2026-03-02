import cv2
from PIL import Image
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
from config import DEVICE, EMBEDDING_BATCH_SIZE, TOP_K_CROPS
from db import get_collection
from state import TrackState
from model import get_multimodal_model, get_clip_model
from logger import get_logger
from utils import _parse_timestamp

logger = get_logger(__name__)

def index_video(video_path: str, video_name: str) -> None:
    model, processor = get_clip_model()
    collection = get_collection()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Index one frame per second to save space
        if frame_idx % int(fps) == 0:
            timestamp = frame_idx / fps
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Create Embedding
            inputs = processor(images=pil_img, return_tensors="pt").to("cuda")
            outputs = model.get_image_features(**inputs)
            image_emb = outputs.detach().cpu().numpy().flatten().tolist()
            
            # Save to ChromaDB
            collection.add(
                embeddings=[image_emb],
                metadatas=[{"timestamp": timestamp, "type": "scene"}],
                ids=[f"frame_{frame_idx}"]
            )
            logger.info(f"Indexed: {timestamp}s")
            
        frame_idx += 1
    cap.release()

def search_index(query_text: str, video_name: str, min_duration: float = 1.0)  -> tuple:
    model, processor = get_clip_model()
    collection = get_collection()

    # Convert query to vector
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(DEVICE)
    text_features = model.get_text_features(**inputs)
    text_features = F.normalize(text_features, p=2, dim=-1)
    query_emb = text_features.detach().cpu().numpy().tolist()[0]

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=10,
        where={"video_name": video_name},
        include=["metadatas", "distances"]
    )

    if not results["ids"][0]:
        logger.warning(f"No results found for query: '{query_text}'")
        return None, None, query_emb

    # Find best result that has a meaningful duration
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]

        start = meta["first_seen"]
        end = meta["last_seen"]

        # Parse timestamps to check duration
        start_sec = _parse_timestamp(start)
        end_sec = _parse_timestamp(end)

        if start_sec is None or end_sec is None:
            continue

        duration = end_sec - start_sec

        if duration >= min_duration:
            logger.info(
                f"Search match: '{query_text}' → track {meta.get('track_id', '?')} "
                f"({start} → {end}, dist={dist:.4f})"
            )
            return start, end, query_emb

        logger.debug(
            f"Skipping result #{i}: track {meta.get('track_id', '?')} "
            f"duration={duration:.1f}s < {min_duration}s ({start} → {end})"
        )

    # Fallback: no result met the duration threshold
    logger.warning(
        f"No result with duration >= {min_duration}s for query: '{query_text}'"
    )
    return None, None, query_emb


def get_average_embedding(crops: list, batch_size=EMBEDDING_BATCH_SIZE) -> list:
    model, processor = get_clip_model()

    all_embeddings = []
    images = [item["image"] for item in crops]
    
    # Process in chunks of 32
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        inputs = processor(images=chunk, return_tensors="pt", padding=True).to(DEVICE)
        
        with torch.no_grad():
            chunk_embeddings = model.get_image_features(**inputs)
            all_embeddings.append(chunk_embeddings.cpu()) # Move to CPU to save VRAM
            
    # Combine and average
    final_stack = torch.cat(all_embeddings, dim=0)
    average_embedding = torch.mean(final_stack, dim=0, keepdim=True)
    average_embedding /= average_embedding.norm(dim=-1, keepdim=True) # Normalize the final average embedding

    return average_embedding.cpu().numpy().flatten().tolist()


def add_collection(video_name: str, embeddings: list, state: TrackState, tid: int) -> None:
    collection = get_collection()

    first_seen = state.first_seen.get(tid, "unknown")
    last_seen = state.last_seen.get(tid, "unknown")

    # Normalize: ensure embeddings is always a list of lists
    # A flat list [0.1, 0.2, ...] means a single embedding → wrap it
    if embeddings and not isinstance(embeddings[0], list):
        embeddings = [embeddings]

    # Store each embedding as a separate entry (same metadata, different IDs)
    ids = []
    metas = []
    for i in range(len(embeddings)):
        ids.append(f"{video_name}_object_{tid}_crop_{i}")
        metas.append({
            "video_name": video_name,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "type": "object",
            "track_id": int(tid),
            "crop_index": i,
        })

    collection.upsert(
        embeddings=embeddings,
        metadatas=metas,
        ids=ids,
    )
    
    logger.info(f"Added object with track ID {tid} to ChromaDB with metadata: {{'video_name': {video_name}, 'first_seen': {first_seen}, 'last_seen': {last_seen}, 'type': 'object'}}")


def create_embeddings_for_crops(state: TrackState) -> None:
    model, processor = get_clip_model()

    for tid, crop in state.buffers.items():
        # Create embedding for the crop
        inputs = processor(images=crop[-1]['image'], return_tensors="pt").to("cuda")
        outputs = model.get_image_features(**inputs)
        normalized_outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        image_emb = normalized_outputs.detach().cpu().numpy().flatten().tolist()
        
        # Store embedding with tid as key
        state.all_embeddings[tid] = image_emb


def create_embeddings_using_multimodel_model(state: TrackState) -> None:

    model, processor = get_multimodal_model()

    for tid, crops in state.buffers.items():
        if len(crops) == 0:
            continue

        images = [item['image'] for item in crops]

        # Sample N evenly spaced frames (model-dependent, usually 8 or 16)
        num_frames = min(8, len(images))
        indices = np.linspace(0, len(images) - 1, num_frames, dtype=int)
        sampled = [np.array(images[i]) for i in indices]

        # Stack as (T, H, W, C) video tensor
        video_array = np.stack(sampled)

        inputs = processor(videos=video_array, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.get_video_features(**inputs)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)

        state.all_embeddings[tid] = outputs.cpu().numpy().flatten().tolist()


def select_best_crops(crops: list, top_k: int = TOP_K_CROPS) -> list:
    """Select top-K crops by confidence, spread across time."""
    if len(crops) <= top_k:
        return crops

    # Sort by confidence descending
    sorted_crops = sorted(crops, key=lambda x: x["confidence"], reverse=True)

    # Take top 2*K by confidence, then spread across time
    candidates = sorted_crops[:top_k * 2]
    candidates.sort(key=lambda x: x["frame_num"])

    # Evenly sample across time
    indices = np.linspace(0, len(candidates) - 1, top_k, dtype=int)
    return [candidates[i] for i in indices]


def get_embeddings_for_storage(crops: list, batch_size=EMBEDDING_BATCH_SIZE) -> list:
    """Return individual embeddings for top-K crops (not averaged)."""
    if not crops:
        return []

    best_crops = select_best_crops(crops)

    model, processor = get_clip_model()
    all_embeddings = []

    images = [crop["image"] for crop in best_crops]

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu())

    stacked = torch.cat(all_embeddings, dim=0)
    return stacked.numpy().tolist()  # list of embeddings


def get_image_embedding_dinov2(model, processor, images, device=DEVICE):
    """DINOv2 image embedding — for Re-ID."""
    if not isinstance(images, list):
        images = [images]

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token embedding
        emb = outputs.last_hidden_state[:, 0, :]
    return F.normalize(emb, p=2, dim=-1)