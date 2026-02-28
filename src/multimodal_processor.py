import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration,BitsAndBytesConfig
import av
import numpy as np

# This '4-bit' config is the magic that makes it fit in your 12GB VRAM
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "LanguageBind/Video-LLaVA-7B-hf"

model = VideoLlavaForConditionalGeneration.from_pretrained(
    model_id,
    # quantization_config=quant_config,
    device_map="auto" # This automatically balances memory
)

processor = VideoLlavaProcessor.from_pretrained(model_id)

def read_video(path, num_frames=8):
    container = av.open(path)
    total_frames = container.streams.video[0].frames
    # Sample 8 frames uniformly across the video
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

# 2. Prepare the Video and Query
video_path = r".\data\clips\short_clip.mp4"
video_clip = read_video(video_path)
query = "USER: <video>\nAt what timestamp does the bus enter the frame?? ASSISTANT:"

# 3. Process and Generate
inputs = processor(text=query, videos=video_clip, return_tensors="pt").to("cuda", torch.float16)
out = model.generate(**inputs, max_new_tokens=100)

print(processor.batch_decode(out, skip_special_tokens=True)[0])