import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mmx_swarm.duality import analyze_conversation

def test_analyze_conversation():
    msgs = [
        {"speaker": "human", "text": "hi"},
        {"speaker": "assistant", "text": "hello"}
    ]
    dual = analyze_conversation(msgs)
    assert dual.dual_vector.human.shape == (99,)
    assert dual.dual_vector.assistant.shape == (99,)
