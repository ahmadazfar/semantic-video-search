import json
import os
from config import *

def save_video_to_path(DIR: str, uploaded_file: object, video_name: str, type: str) -> str:
    file_path = os.path.join(DIR, video_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if type == "annotated":
        save_path_to_registry(video_name + "_annotated", file_path)

    elif type == "raw":
        save_path_to_registry(video_name + "_raw", file_path)

    return file_path

def save_path_to_registry(video_name: str, file_path: str):
    # Load existing data
    if os.path.exists(VIDEO_REGISTRY_PATH):
        with open(VIDEO_REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    # Add/Update the entry
    registry[video_name] = file_path

    # Save back to file
    with open(VIDEO_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

def get_path_from_registry(video_name: str, type: str) -> str:
    if not os.path.exists(VIDEO_REGISTRY_PATH):
        return None
    
    with open(VIDEO_REGISTRY_PATH, "r") as f:
        registry = json.load(f)
    
    # Return the path if it exists, else return None
    return registry.get(video_name + f"_{type}", None)