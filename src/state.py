from dataclasses import dataclass, field


@dataclass
class TrackState:
    """Holds all per-track state for a single video processing run."""
    buffers: dict = field(default_factory=dict)       # tid -> list of crop data
    stationary: dict = field(default_factory=dict)     # tid -> last bbox
    stationary_count: dict = field(default_factory=dict)  # tid -> int
    lost_counts: dict = field(default_factory=dict)    # tid -> int
    all_embeddings: dict = field(default_factory=dict)   # tid -> embedding list
    first_seen: dict = field(default_factory=dict)   # tid -> timestamp string
    last_seen: dict = field(default_factory=dict)   # tid -> timestamp string
    pending_finalize: dict = field(default_factory=dict)  # tid -> {"age": int, "embedding": list}