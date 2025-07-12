from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class DualPersonalityVector:
    human: np.ndarray
    assistant: np.ndarray

@dataclass
class ConversationDuality:
    human_patterns: Dict[str, Any]
    assistant_patterns: Dict[str, Any]
    dual_vector: DualPersonalityVector


def analyze_conversation(messages: List[Dict[str, Any]]) -> ConversationDuality:
    human_msgs = [m["text"] for m in messages if m.get("speaker") == "human"]
    ai_msgs = [m["text"] for m in messages if m.get("speaker") == "assistant"]
    def make_vec(texts):
        joined = " ".join(texts)
        seed = abs(hash(joined)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(99)
    human_vec = make_vec(human_msgs)
    ai_vec = make_vec(ai_msgs)
    dual = DualPersonalityVector(human_vec, ai_vec)
    patterns = {
        "count": len(messages),
        "human_count": len(human_msgs),
        "assistant_count": len(ai_msgs),
    }
    return ConversationDuality(patterns, patterns, dual)
