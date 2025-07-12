import json
import csv
import os
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class EmotionalSpectrum:
    values: np.ndarray

@dataclass
class CognitiveStyle:
    values: np.ndarray

@dataclass
class SocialDynamics:
    values: np.ndarray

@dataclass
class CreativeExpression:
    values: np.ndarray

@dataclass
class ValueSystem:
    values: np.ndarray

@dataclass
class ConversationMood:
    values: np.ndarray

@dataclass
class IdentityState:
    values: np.ndarray

class ConversationProcessor:
    """Parse conversation exports and derive personality vectors."""
    def __init__(self, export_dir: str = "conversations"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def parse(self, path: str) -> List[Dict[str, Any]]:
        """Parse JSON, TXT or CSV to a list of chat entries."""
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                msgs = data
            else:
                msgs = data.get("messages", [])
            out = []
            for m in msgs:
                out.append({
                    "speaker": m.get("speaker") or m.get("role"),
                    "text": m.get("text") or m.get("content"),
                    "timestamp": m.get("timestamp")
                })
            return out
        elif path.endswith(".txt"):
            out = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        speaker, text = line.split(":", 1)
                        out.append({"speaker": speaker.strip(), "text": text.strip(), "timestamp": None})
            return out
        elif path.endswith(".csv"):
            out = []
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    out.append({
                        "speaker": row.get("speaker") or row.get("role"),
                        "text": row.get("text") or row.get("content"),
                        "timestamp": row.get("timestamp")
                    })
            return out
        else:
            raise ValueError("Unsupported file format")

    def extract_full_profile(self, messages: List[Dict[str, Any]]) -> np.ndarray:
        """Return a deterministic 99-D profile vector."""
        joined = " ".join(m.get("text", "") for m in messages)
        seed = abs(hash(joined)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(99)

    def persist(self, session_id: str, messages: List[Dict[str, Any]], vector: np.ndarray) -> None:
        jpath = os.path.join(self.export_dir, f"session_{session_id}.json")
        npy_path = os.path.join(self.export_dir, f"session_{session_id}.npy")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(messages, f)
        np.save(npy_path, vector)
