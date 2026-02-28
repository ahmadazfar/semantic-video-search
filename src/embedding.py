import cv2
from PIL import Image
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
from config import DEVICE, EMBEDDING_BATCH_SIZE
from db import get_collection
from state import TrackState
from model import get_multimodal_model, get_clip_model

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
            print(f"Indexed: {timestamp}s")
            
        frame_idx += 1
    cap.release()

def search_index(query_text: str, video_name: str)  -> tuple:
    model, processor = get_clip_model()
    collection = get_collection()

    # Convert query to vector
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(DEVICE)
    text_features = model.get_text_features(**inputs)
    text_features = F.normalize(text_features, p=2, dim=-1)
    query_emb = text_features.detach().cpu().numpy().tolist()[0]

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=1,
        where={"video_name": video_name},
        include=["metadatas", "distances"]
    )

    # Extract the timestamps
    for i in range(len(results['ids'][0])):
        best_match_meta = results['metadatas'][0][i] # Get first result's metadata
        dist = results['distances'][0][i]
        
        start = best_match_meta['first_seen']
        end = best_match_meta['last_seen']
        
        # print(f"Match Found! (Distance: {dist:.4f})")
        # print(f"Object first appeared at: {start} seconds")
        # print(f"Object disappeared at: {end} seconds")

        return start, end, query_emb


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


def add_collection(video_name: str, embedding: list, track_buffers: dict, tid: int) -> None:
    collection = get_collection()

    buffer_data = track_buffers.get(tid, [])
    
    timestamps = [item["timestamp"] for item in buffer_data]

    unique_id = f"{video_name}_object_{tid}"

    collection.add(
                embeddings=[embedding],
                metadatas=[{"video_name":video_name,"first_seen": min(timestamps), "last_seen": max(timestamps),"type": "object"}],
                ids=[unique_id]
            )
    
    return print("Added object with track ID", tid, "to ChromaDB with metadata:", {"video_name":video_name,"first_seen": min(timestamps), "last_seen": max(timestamps),"type": "object"})


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


