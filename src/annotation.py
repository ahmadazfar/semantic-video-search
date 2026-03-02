import os
import cv2
import time
import numpy as np
from PIL import Image
import supervision as sv
from rfdetr.util.coco_classes import COCO_CLASSES
from  config import *
from db import *
from model import get_detector
from supervision import ByteTrack
from embedding import *
from tracking import crop_object, flow_update
from state import TrackState
from utils import get_timestamp
from reid import ReIDTracker
from logger import get_logger

logger = get_logger(__name__)

def detect_objects_and_annotate(video_path: str, video_name: str) -> str:

    # Load Model
    rfdetr_model = get_detector()
    # Tracker
    tracker = ByteTrack(lost_track_buffer=LOST_TRACK_BUFFER,track_activation_threshold=0.25,   # lower = keep more tentative tracks
        minimum_matching_threshold=0.7,     # lower = more lenient matching
        frame_rate=30   )
    reid_tracker = ReIDTracker(tracker, reid_threshold=REID_THRESHOLD, max_lost_age=REID_MAX_LOST_AGE)

    # State for tracking
    state = TrackState()
    # Annotators
    color = sv.ColorPalette.from_hex(COLOR_PALETTE_HEX)

    output_path = os.path.join(ANNOTATED_VIDEOS_DIR, f"{video_name}_annotated.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # output_with_audio_path = os.path.join(ANNOTATED_VIDEOS_DIR, f"{video_name}_annotated_with_audio.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Measure processing time
    start_time = time.time()

    #Initialize optical flow with the first frame
    ret, frame_prev = cap.read()
    #For optical flow, we need grayscale
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    detections = rfdetr_model.predict(frame_prev)
    centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
    points_prev = centers.reshape(-1, 1, 2).astype(np.float32)
    final_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_num = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        logger.debug(f"Processing frame {frame_num}/{final_frame}")
        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        # For optical flow, we need grayscale
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

        bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            color=color,
            text_color=sv.Color.BLACK,
            text_scale=LABEL_TEXT_SCALE,
            text_thickness=LABEL_TEXT_THICKNESS,
            text_padding=LABEL_TEXT_PADDING,
            smart_position=True
        )
        # Detect objects
        if frame_num % DETECTION_INTERVAL == 0 or frame_num == 1:
            # Expensive detection
            detections = rfdetr_model.predict(image, threshold=0.5)
            mask = np.isin(detections.class_id, [1]) # Only person class for now to save time, can expand later
            detections = detections[mask]
            centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            points_prev = centers.reshape(-1, 1, 2).astype(np.float32)
            tracked = tracker.update_with_detections(detections)
            for i, tid in enumerate(tracked.tracker_id):
                if tid in reid_tracker.id_mapping:
                    tracked.tracker_id[i] = reid_tracker.resolve_id(tid)
        else:
            # Cheap tracking with optical flow
            points_prev, detections = flow_update(gray_prev, gray_curr, points_prev, detections, tracker)

        gray_prev = gray_curr.copy()

        #Labels from RF-DETR
        labels = [
            f"{COCO_CLASSES[class_id]} {conf:.2f} ID:{tid}"
            for class_id, conf, tid in zip(
                tracked.class_id, tracked.confidence, tracked.tracker_id
            )
        ]

        current_ids = set(tracked.tracker_id)

        # Check for lost tracks and finalize their indexing if they've been lost for too long
        current_timestamp = get_timestamp(frame_num, fps)

        
        # For all previously seen track IDs, if they are NOT in the current frame, increment their lost count by 1
        all_known_ids = list(state.lost_counts.keys())
        for tid in all_known_ids:
            if tid not in current_ids:
                state.lost_counts[tid] += 1
                reid_tracker.mark_lost(tid)  # Mark this track ID as lost in the ReID tracker
            #THE "LOST" TRIGGER  
            if state.lost_counts[tid] == LOST_FRAME_THRESHOLD:
                # Move the buffer to pending finalize state
                state.pending_finalize[tid] = {
                    "age": 0,
                }

        # Crop every N frames (N = fps / divisor)
        if frame_num % int(fps/CROP_INTERVAL_DIVISOR) == 0 or frame_num == 1:
            current_ids = set()
            logger.info(f"Processing crops for frame {frame_num}")
            crop_object(tracked, frame, frame_num, state, fps, video_name, reid_tracker, current_ids)



        # Update first_seen / last_seen for EVERY frame the object is visible
        for tid in current_ids:
            state.lost_counts[tid] = 0
            if tid not in state.first_seen:
                state.first_seen[tid] = current_timestamp
            state.last_seen[tid] = current_timestamp  # always update
            
            # If this track was pending finalization, cancel it (it came back!)
            if tid in state.pending_finalize:
                logger.info(f"Track {tid} came back — cancelling pending finalization")
                del state.pending_finalize[tid]





        expired_pending = []
        for tid in list(state.pending_finalize.keys()):
            state.pending_finalize[tid]["age"] += 1
            # Grace period: same as ReID max_lost_age
            if state.pending_finalize[tid]["age"] >= REID_MAX_LOST_AGE:
                expired_pending.append(tid)

        for tid in expired_pending:
            if tid in state.buffers and state.buffers[tid] != []:  # Only finalize if we have buffered crops/embeddings or if we're at the final frame
                logger.info(f"Finalizing track {tid} after {REID_MAX_LOST_AGE} frames in pending state")
                # Here we would finalize the embedding and add to ChromaDB if we had a pending embedding
                # For now, we just clean up the pending state
                state.pending_finalize.pop(tid, None)
                emb_for_top_crops = get_embeddings_for_storage(state.buffers[tid], EMBEDDING_BATCH_SIZE)
                add_collection(video_name, emb_for_top_crops, state, tid)
                
                #Clean memory
                reid_tracker.track_embeddings.pop(tid, None)  
                reid_tracker.lost_tracks.pop(tid, None)
                state.lost_counts.pop(tid, None)
                state.buffers.pop(tid, None)
                state.first_seen.pop(tid, None)
                state.last_seen.pop(tid, None)
        
        if frame_num == final_frame:
            logger.info("Finalizing all tracks at final frame")
            for tid in list(state.buffers.keys()):
                if state.buffers[tid] != []: 
                    emb_for_top_crops = get_embeddings_for_storage(state.buffers[tid], EMBEDDING_BATCH_SIZE)
                    add_collection(video_name, emb_for_top_crops, state, tid)
                    
                    #Clean memory
                    reid_tracker.track_embeddings.pop(tid, None)  
                    reid_tracker.lost_tracks.pop(tid, None)
                    state.lost_counts.pop(tid, None)
                    state.buffers.pop(tid, None)
                    state.first_seen.pop(tid, None)
                    state.last_seen.pop(tid, None)
                    state.pending_finalize.pop(tid, None)

        reid_tracker.age_lost_tracks()

        # Annotate
        annotated = frame.copy()
        annotated = bbox_annotator.annotate(annotated, tracked)
        annotated = label_annotator.annotate(annotated, tracked, labels)

        # Uncomment for real time tracking
        cv2.imshow("RF-DETR Real-Time Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out.write(annotated)
            break

        # Write annotated frame to output
        out.write(annotated)

        frame_num += 1

        # Print progress every 30 frames
        # if frame_num % 30 == 0 :
        #     frame_end_time = time.time()
        #     elapsed_time = frame_end_time - frame_start_time
        #     print(f"Processed {frame_num} frames... Elapsed time for last 30 frames: {elapsed_time:.4f} seconds")

        #     frame_start_time = time.time()
    cap.release()
    out.release()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Done! Annotated video saved to {output_path}")
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    return output_path
    # cv2.destroyAllWindows()
    # merge_audio(video_path, output_path, output_with_audio_path)

