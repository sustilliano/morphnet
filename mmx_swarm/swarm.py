import numpy as np
from typing import List
from .bias_mixer import get_global_bias
from .duality import DualPersonalityVector
from .physics import EnhancedPersonalityTensorPhysics

class SwarmIdentity:
    def __init__(self, name: str, vector: np.ndarray, dual: DualPersonalityVector = None):
        self.name = name
        self.vector = np.asarray(vector, dtype=float)
        self.dual_vector = dual

class SwarmConsciousness:
    def __init__(self):
        self.identities: List[SwarmIdentity] = []
        self.physics = EnhancedPersonalityTensorPhysics()

    def add_identity(self, name: str, vector: np.ndarray, dual: DualPersonalityVector = None) -> SwarmIdentity:
        ident = SwarmIdentity(name, vector, dual)
        self.identities.append(ident)
        self.physics.register_dual_personality(name, vector)
        return ident

    def swarm_update(self) -> None:
        self.physics.update()

    def generate_collective_response(self, query: str):
        if not self.identities:
            return {
                "collective_vector": np.zeros(99),
                "top_contributors": [],
                "consensus": "none",
                "clusters": [],
            }
        base = np.mean([i.vector for i in self.identities], axis=0)
        bias = get_global_bias()
        final = 0.5 * base + 0.5 * bias
        return {
            "collective_vector": final,
            "top_contributors": [i.name for i in self.identities[:3]],
            "consensus": "approx",
            "clusters": [0 for _ in self.identities],
        }
