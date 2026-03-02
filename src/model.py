from rfdetr import RFDETRMedium
from transformers import CLIPModel, CLIPProcessor, VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from supervision import ByteTrack
from config import DEVICE, RFDETR_RESOLUTION, CLIP_MODEL_NAME, VIDEO_LLAVA_MODEL_NAME, DINOV2_MODEL_NAME
from transformers import AutoModel, AutoImageProcessor


_clip_model = None
_clip_processor = None
_detector = None
_multimodal_model = None
_multimodal_processor = None
_dinov2_model = None
_dinov2_processor = None


def get_clip_model():
    """Load CLIP model and processor for embedding. Singleton."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _clip_model, _clip_processor


def get_detector():
    """Load RF-DETR object detection model. Singleton."""
    global _detector
    if _detector is None:
        _detector = RFDETRMedium(resolution=RFDETR_RESOLUTION, device=DEVICE)
        _detector.optimize_for_inference()
    return _detector

def get_multimodal_model():
    """Load VideoLLaVA multimodal model and processor. Singleton."""
    global _multimodal_model, _multimodal_processor
    if _multimodal_model is None:
        _multimodal_processor = VideoLlavaProcessor.from_pretrained(VIDEO_LLAVA_MODEL_NAME)
        _multimodal_model = VideoLlavaForConditionalGeneration.from_pretrained(VIDEO_LLAVA_MODEL_NAME).to(DEVICE)
    return _multimodal_model, _multimodal_processor

def get_dinov2_model():
    """DINOv2 model — used for Re-ID (instance-level matching)."""
    global _dinov2_model, _dinov2_processor
    if _dinov2_model is None:
        _dinov2_model = AutoModel.from_pretrained(DINOV2_MODEL_NAME).to(DEVICE)
        _dinov2_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME)
        _dinov2_model.eval()
    return _dinov2_model, _dinov2_processor
