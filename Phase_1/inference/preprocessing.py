import torch 
from decord import VideoReader
from decord import cpu, gpu
from PIL import Image

def preprocessing(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(list(range(0, len(vr), 15))).asnumpy()
    pil_frames = [Image.fromarray(frames[i]) for i in range(len(frames))]
    return pil_frames