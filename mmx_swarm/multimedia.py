import numpy as np
from .swarm import SwarmConsciousness

try:
    from transformers import CLIPProcessor, CLIPModel
except Exception:  # avoid heavy deps if unavailable
    CLIPProcessor = None
    CLIPModel = None

try:
    import whisper
except Exception:
    whisper = None


def add_image_identity(path: str, swarm: SwarmConsciousness) -> str:
    if CLIPModel is None:
        vec512 = np.random.random(512)
    else:
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        from PIL import Image
        image = Image.open(path)
        inputs = processor(images=image, return_tensors='pt')
        with torch.no_grad():
            vec512 = model.get_image_features(**inputs).numpy().flatten()
    vec99 = vec512[:99]
    swarm.add_identity(path, vec99)
    return path


def transcribe_audio(path: str) -> str:
    if whisper is None:
        return ""
    model = whisper.load_model('base')
    result = model.transcribe(path)
    return result['text']
