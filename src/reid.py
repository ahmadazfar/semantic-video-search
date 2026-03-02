import numpy as np
import torch
from config import DEVICE, REID_THRESHOLD, REID_MAX_LOST_AGE
from model import get_clip_model
from state import TrackState
from logger import get_logger
from embedding import get_image_embedding_dinov2
from model import get_dinov2_model

logger = get_logger(__name__)


class ReIDTracker:
    """Wraps ByteTrack with appearance-based re-identification."""

    def __init__(self, tracker, reid_threshold=REID_THRESHOLD, max_lost_age=REID_MAX_LOST_AGE):
        self.tracker = tracker
        self.reid_threshold = reid_threshold
        self.max_lost_age = max_lost_age

        self.track_embeddings = {}   # tid -> embedding (numpy)
        self.lost_tracks = {}        # old_tid -> {"embedding": ..., "age": int}
        self.id_mapping = {}         # new_tid -> original_tid

    def update_with_detections(self, detections):
        tracked = self.tracker.update_with_detections(detections)
        return tracked

    def update_embedding(self, tid, crop_image):
        """Update appearance embedding using DINOv2."""
        model, processor = get_dinov2_model()
        emb = get_image_embedding_dinov2(model, processor, crop_image)
        self.track_embeddings[tid] = emb.cpu().numpy().flatten()

    def resolve_id(self, tid):
        """Get the original ID for a track (follows remapping chain)."""
        resolved_tid = self.id_mapping.get(tid, tid)
        logger.debug(f"Resolved track ID {tid} to {resolved_tid} using Re-ID mapping.")
        return resolved_tid

    def check_reappearance(self, new_tid):
        """
        Check if a new track matches any recently lost track.
        Returns the original tid if matched, otherwise the new tid.
        """

        new_emb = self.track_embeddings[new_tid]
        best_match = None
        best_score = -1

        for old_tid, lost_data in self.lost_tracks.items():
            old_emb = lost_data["embedding"]
            score = np.dot(new_emb, old_emb)
            if score > best_score:
                best_score = score
                best_match = old_tid

        if best_match is not None and best_score >= self.reid_threshold:
            logger.info(f"Re-ID: track {new_tid} -> restored as track {best_match} (score={best_score:.3f})")
            # Transfer new embedding to old ID
            self.track_embeddings[best_match] = self.track_embeddings.pop(new_tid)

            # Remove from lost pool — it's back
            del self.lost_tracks[best_match]

            # Map new_tid → old_tid so annotation loop uses the old ID
            self.id_mapping[new_tid] = best_match


            return best_match

        return new_tid

    def mark_lost(self, tid):
        """Move a track to the lost pool for future re-id."""
        if tid in self.track_embeddings and tid not in self.lost_tracks:
            self.lost_tracks[tid] = {
                "embedding": self.track_embeddings[tid],
                "age": 0,
            }
            logger.info(f"Marked track {tid} as lost and added to Re-ID pool.")

    def age_lost_tracks(self):
        """Increment age of lost tracks, remove expired ones."""
        expired = []
        for tid in self.lost_tracks:
            self.lost_tracks[tid]["age"] += 1
            if self.lost_tracks[tid]["age"] > self.max_lost_age:
                expired.append(tid)
        for tid in expired:
            del self.lost_tracks[tid]