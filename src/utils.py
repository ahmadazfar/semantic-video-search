import cv2
import numpy as np
from config import CROP_PADDING_PERCENT, CROP_TARGET_SIZE


def get_timestamp(frame_num: int, fps: float) -> str:
    total_seconds = frame_num / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def timestamp_to_seconds(ts_str: str) -> int:
    # Split by colon: "0:00:24" -> ["0", "00", "24"]
    parts = ts_str.split(':')
    if len(parts) == 3:  # HH:MM:SS
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:  # MM:SS
        m, s = map(int, parts)
        return m * 60 + s
    return int(parts[0]) # Just seconds


def get_padded_bbox(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, padding_percent=CROP_PADDING_PERCENT) -> tuple: 
    w = x2 - x1
    h = y2 - y1
    pad_x = w * padding_percent
    pad_y = h * padding_percent
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(img_w, int(x2 + pad_x))
    y2 = min(img_h, int(y2 + pad_y))
    return x1, y1, x2, y2

def resize_with_padding(image: np.ndarray, target_size=CROP_TARGET_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    
    # Determine scale factor to fit the longest side
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize keeping aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black square canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Center the resized image on the canvas
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas