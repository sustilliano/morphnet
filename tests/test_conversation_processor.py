import json
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mmx_swarm.conversation_processor import ConversationProcessor

def test_parse_and_profile(tmp_path):
    data = [
        {"speaker": "human", "text": "hi"},
        {"speaker": "assistant", "text": "hello"}
    ]
    p = tmp_path/'conv.json'
    p.write_text(json.dumps(data))
    cp = ConversationProcessor(export_dir=tmp_path)
    msgs = cp.parse(str(p))
    assert len(msgs) == 2
    vec = cp.extract_full_profile(msgs)
    assert vec.shape == (99,)
    cp.persist('001', msgs, vec)
    assert (tmp_path/'session_001.npy').exists()
