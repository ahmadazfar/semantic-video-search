import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from annotation import *
from embedding import *
from queried_detection import run_detection_on_timestamp
from landing import render_landing_page
from db import get_indexed_videos
from video_manager import *
from logger import get_logger

logger = get_logger(__name__)

# Assuming your functions are in 'logic.py'
# from logic import run_indexing, run_top_k_detection, load_models
st.set_page_config(page_title="Video Search AI", layout="wide", page_icon="🔍")

page = st.sidebar.selectbox("Navigate", ["🏠 Home", "Upload & Index", "Search Indexed Videos", "Object Detection"])


video_list = get_indexed_videos()


if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None

if 'video_path' not in st.session_state:
    st.session_state.video_path = None

# --- HOME / LANDING PAGE ---
if page == "🏠 Home":
    render_landing_page()

# --- PAGE 1: UPLOAD & INDEX ---
elif page == "Upload & Index":

    st.header("Create a New Index")
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi"])
    video_name = st.text_input("Name this index", placeholder="e.g., Highway_Camera_1")

    if uploaded_file and video_name and st.button("Start Backend Indexing"):
        with st.spinner("Processing video..."):
            file_path = save_video_to_path(UPLOADED_VIDEOS_DIR, uploaded_file, video_name, "raw")
            # Save the result to local disk
            annotated_video_path = detect_objects_and_annotate(file_path, video_name)
            save_path_to_registry(video_name + "_annotated", annotated_video_path,)
            
             # Update the video list for the search page        
            st.success(f"Index for '{video_name}' saved successfully!")

elif page == "Search Indexed Videos":
    st.header("Search & Annotate")
    # Initialize session state if it doesn't exist
    if "processed_videos" not in st.session_state:
        st.session_state.processed_videos = []

    if not video_list:
        st.warning("No indexed videos found.")
    else:
        selected_videos = st.multiselect("Select indexed videos", video_list)
        query_text = st.text_input("Enter your search query")

        if query_text and st.button("Run Search & Detection"):
            st.session_state.processed_videos = []
            for selected_video in selected_videos:
                with st.spinner(f"Searching {selected_video}..."):
                    start, end, query_emb = search_index(query_text, selected_video)
                    video_path = get_path_from_registry(selected_video, "raw")
                    actual_output = run_detection_on_timestamp(video_path, start, end, query_emb, selected_video)
                    st.session_state.processed_videos.append({
                        "video_name": selected_video,
                        "path": actual_output,
                        "start": start,
                        "end": end,
                    })
            st.success("Search completed!")

    # Display all processed videos
    for item in st.session_state.processed_videos:
        st.subheader(f"📹 {item['video_name']}")
        st.write(f"Segment: {item['start']} → {item['end']}")
        abs_path = os.path.abspath(item["path"])
        if os.path.exists(abs_path):
            with open(abs_path, "rb") as v_file:
                st.video(v_file.read(), format="video/mp4")
        else:
            st.error(f"Video file not found: {abs_path}")

elif page == "Object Detection":
    st.header("Track Objects in a Video")
    if not video_list:
        st.warning("No indexed videos found.")
    else:
        selected_video = st.selectbox("Select an indexed video for object detection", video_list)
        video_path = get_path_from_registry(selected_video, "annotated")
        logger.info(f"Selected video path for annotation: {video_path}")
        if os.path.exists(video_path):
            with open(video_path, "rb") as v_file:
                st.video(v_file.read(), format="video/mp4")
        else:
            st.error(f"Video file not found: {video_path}")