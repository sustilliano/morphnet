import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mmx_swarm.physics import PersonalityTensorPhysics

def test_register_and_update():
    phys = PersonalityTensorPhysics()
    phys.register_dual_personality('id1', np.zeros(99))
    assert 'id1' in phys.positions
    pos_before = phys.positions['id1'].copy()
    phys.update()
    assert phys.positions['id1'].shape == (3,)
    assert not np.allclose(phys.positions['id1'], pos_before) or np.allclose(phys.positions['id1'], pos_before)
