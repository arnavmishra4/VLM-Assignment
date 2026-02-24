import json
import re
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch 
from decord import VideoReader, cpu
from PIL import Image
import numpy as np

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

def prediction(frames: list) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": """Analyze this warehouse packaging video clip.
Respond ONLY with a raw JSON object exactly like this example, no explanation, no markdown:

{"clip_id": "unknown", "dominant_operation": "Tape", "temporal_segment": {"start_frame": 14, "end_frame": 98}, "anticipated_next_operation": "Put Items", "confidence": 0.87}

dominant_operation must be one of: Box Setup, Inner Packing, Tape, Put Items, Pack, Wrap, Label, Final Check, Idle, Unknown
anticipated_next_operation must be one of the same list."""}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200)
    raw_output = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])

    json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    else:
        return {"error": "model did not return valid JSON", "raw": raw_output}


def predict_batch(video_frames_list: list) -> list:
    """Pass a list of frame lists, get back a list of predictions."""
    return [prediction(frames) for frames in video_frames_list]