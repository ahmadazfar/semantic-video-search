import os
import cv2
import numpy as np
from config import (
    CROPPED_OBJECTS_DIR, STATIONARY_THRESHOLD_PX, STATIONARY_CHECK_COUNT
)
from utils import get_timestamp, get_padded_bbox, resize_with_padding
from embedding import get_average_embedding, add_collection
import supervision as sv
from supervision import ByteTrack
from state import TrackState


def crop_object(tracked: sv.Detections, frame: np.ndarray, frame_num: int, state: TrackState, fps: float, video_name: str) -> None:
    base_folder = os.path.join(CROPPED_OBJECTS_DIR, video_name)
    os.makedirs(base_folder, exist_ok=True)
    
    timestamp = get_timestamp(frame_num, fps)

    # Ensure tracked has data to avoid errors on empty frames
    if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
        return

    for box, tid, conf in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
        x1, y1, x2, y2 = map(int, box)

        # 1. Boundary Clipping (Prevents crashes if box is off-screen)
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        # 2. Add padding to the bounding box (20% of the box size) while ensuring we don't go out of frame boundaries
        x1, y1, x2, y2 = get_padded_bbox(x1, y1, x2, y2, w, h, padding_percent=0.2)
        
        # 3. Slice the frame
        cropped = frame[y1:y2, x1:x2].copy()

        # 4. Resize with padding to 224x224 (Maintains aspect ratio and adds black padding if needed)
        processed_crop = resize_with_padding(cropped, target_size=224)

        if is_stationary(tid, box, state, threshold=STATIONARY_THRESHOLD_PX):
            if state.stationary_count[tid] == STATIONARY_CHECK_COUNT:  # If we've just reached the threshold for being stationary
                ave_embedding = get_average_embedding(state.buffers[tid])
                add_collection(video_name, ave_embedding, state.buffers, tid)
                return
            else:
                return  # Skip cropping and embedding for this frame since it's stationary and has already been embedded once

            
        if tid not in state.buffers:
            state.buffers[tid] = []

        if processed_crop.size > 0:
            # file_path = f"{base_folder}/track_id_{tid}_{conf:.2f}conf_{frame_num}.jpg"
            state.buffers[tid].append({
                "image": processed_crop,
                "confidence": conf,
                "frame_num": frame_num,
                "timestamp": timestamp
            })
            # cv2.imwrite(file_path, processed_crop)


def is_stationary(tid: int, current_box: np.ndarray, state: TrackState, threshold=STATIONARY_THRESHOLD_PX) -> bool:
    """
    Checks if the object has moved more than 'threshold' pixels 
    from its last recorded position in the buffer.
    """
    if tid not in state.stationary:
        state.stationary[tid] = current_box # Initialize stationary box for this track ID
        return False
        
    last_box = state.stationary[tid] # Get the last saved box


    # Calculate Euclidean distance between centers
    last_center = ((last_box[0]+last_box[2])/2, (last_box[1]+last_box[3])/2)
    curr_center = ((current_box[0]+current_box[2])/2, (current_box[1]+current_box[3])/2)
    
    dist = ((last_center[0] - curr_center[0])**2 + (last_center[1] - curr_center[1])**2)**0.5

    if dist < threshold:
        state.stationary_count[tid] = state.stationary_count.get(tid, 0) + 1
        if state.stationary_count[tid] >= STATIONARY_CHECK_COUNT:  # If stationary for 10 consecutive checks (5 seconds at half-second intervals)
            print(f"Track {tid} is stationary for 300 frames (moved {dist:.2f} pixels). Skipping crop.")
            return True
        else:
            print(f"Track {tid} is stationary (moved {dist:.2f} pixels). Count: {state.stationary_count[tid]}")
            return False

    else:
        state.stationary[tid] = current_box # Update the reference box to the new position so we can track movement from HERE
        return False


def flow_update(gray_prev: np.ndarray, gray_curr: np.ndarray, points_prev: np.ndarray, detections: sv.Detections, tracker: ByteTrack) -> tuple:
    """
    Use optical flow to shift detections between full detection frames.

    Returns:
        tracked: Updated tracker detections.
        points_prev: Updated points for next frame (or None if lost).
    """
    # No points from previous frame
    if points_prev is None or len(points_prev) == 0:
        tracked = tracker.update_with_detections(sv.Detections.empty())
        return tracked, None, detections

    points_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_curr, points_prev, None
    )

    # Optical flow failed completely
    if status is None:
        tracked = tracker.update_with_detections(sv.Detections.empty())
        return tracked, None, detections

    mask = status.flatten().astype(bool)
    points_curr = points_curr[mask]
    points_prev = points_prev[mask]
    detections = detections[mask]

    # No surviving points
    if len(detections) == 0:
        tracked = tracker.update_with_detections(sv.Detections.empty())
        return tracked, None, detections

    # Shift bounding boxes by the movement delta
    movement = (points_curr - points_prev).reshape(-1, 2)
    detections.xyxy[:, [0, 2]] += movement[:, 0:1]
    detections.xyxy[:, [1, 3]] += movement[:, 1:2]

    tracked = tracker.update_with_detections(detections)
    points_prev = points_curr.reshape(-1, 1, 2)

    return tracked, points_prev, detections 