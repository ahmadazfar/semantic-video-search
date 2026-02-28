import cv2
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from annotation import *
from embedding import create_embeddings_for_crops
from config import COLOR_PALETTE_HEX, LABEL_TEXT_SCALE, LABEL_TEXT_THICKNESS, LABEL_TEXT_PADDING, DETECTION_INTERVAL, LOST_TRACK_BUFFER
from model import get_detector
from utils import timestamp_to_seconds
from db import get_collection
import supervision as sv


def run_detection_on_timestamp(video_path, start_seconds, end_seconds, query_emb, video_name):

    # Annotators
    color = sv.ColorPalette.from_hex(COLOR_PALETTE_HEX)

    # Get Model
    model = get_detector()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Seek directly to the starting frame
    start_seconds = timestamp_to_seconds(start_seconds)
    end_seconds = timestamp_to_seconds(end_seconds)
    start_frame = int(start_seconds * fps) 
    end_frame = int(end_seconds * fps)
    current_frame = start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    state = TrackState()

    output_path = os.path.join(QUERIED_DETECTIONS_DIR, f"{video_name}_queried_detection.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tracker = ByteTrack(lost_track_buffer=LOST_TRACK_BUFFER,track_activation_threshold=0.25,   # lower = keep more tentative tracks
        minimum_matching_threshold=0.7,     # lower = more lenient matching
        frame_rate=30   )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #Initialize optical flow with the first frame
    ret, frame_prev = cap.read()
    current = cap.get(cv2.CAP_PROP_POS_FRAMES)

    #For optical flow, we need grayscale
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    detections = model.predict(frame_prev)
    centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
    points_prev = centers.reshape(-1, 1, 2).astype(np.float32)

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # For optical flow, we need grayscale
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(rgb_frame)
            
        # Create Annotators (recreate each frame in case resolution changes)
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

        if current_frame % DETECTION_INTERVAL == 0: # Do full detection every N frames to correct for drift
            # Expensive detection
            detections = model.predict(image, threshold=0.5)
            centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            points_prev = centers.reshape(-1, 1, 2).astype(np.float32)
            tracked = tracker.update_with_detections(detections)
        
        else: # Use optical flow to shift detections in between
            tracked, points_prev, detections = flow_update(gray_prev, gray_curr, points_prev, detections, tracker)

        gray_prev = gray_curr.copy()

        #Labels from RF-DETR
        labels = [
            f"{COCO_CLASSES[class_id]} {conf:.2f} ID:{tid}"
            for class_id, conf, tid in zip(
                tracked.class_id, tracked.confidence, tracked.tracker_id
            )
        ]

        # Crop every N frames (N = fps / divisor)
        if current_frame % int(fps/CROP_INTERVAL_DIVISOR) == 0 or current_frame == start_frame:
            crop_object(tracked, frame, current_frame, state, fps, video_name) 
            create_embeddings_for_crops(state)
            top_k_match = find_and_target_object(query_emb, state.all_embeddings,1)

        # 4️⃣ Annotate & Save
        just_ids = [item['tid'] for item in top_k_match]
        score_map = {item['tid']: item['score'] for item in top_k_match}
        mask = np.array([tid in just_ids for tid in tracked.tracker_id])
        top_tracked = tracked[mask]

        top_labels = []
        for tid in top_tracked.tracker_id:
            score = score_map.get(tid, 0.0)
            top_labels.append(f"ID:{tid} Score:{score:.2f}")
        
        annotated = frame.copy()
        if len(top_tracked) > 0:
            annotated = bbox_annotator.annotate(scene=annotated, detections=top_tracked)
            annotated = label_annotator.annotate(scene=annotated, detections=top_tracked, labels=top_labels)

        out.write(annotated)

        current_frame += 1
    out.release()
    cap.release()

    # output_path = convert_to_h264(output_path)
    return output_path


def find_and_target_object(query_vec, all_embeddings, k=1):
    
    tids = list(all_embeddings.keys())

    crop_vec = np.array(list(all_embeddings.values()))

    query_vec = np.array(query_vec)

    # Calculate Cosine Similarity (Dot Product)
    similarities = crop_vec @ query_vec

    top_indices = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in top_indices:
        results.append({
            "tid": tids[idx],
            "score": float(similarities[idx])
        })
    return results

