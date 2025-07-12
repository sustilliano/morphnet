import numpy as np

class PersonalityTensorPhysics:
    """Simple physics engine for personality tensors."""
    def __init__(self):
        self.positions = {}
        self.vectors = {}

    def register_dual_personality(self, identity_id: str, vector: np.ndarray) -> None:
        self.positions[identity_id] = np.zeros(3)
        self.vectors[identity_id] = vector

    def update(self) -> None:
        # placeholder update - apply small random walk
        for k in self.positions:
            self.positions[k] += np.random.normal(scale=0.01, size=3)

    def apply_external_stimulus(self, identity_id: str, stimulus: np.ndarray) -> None:
        if identity_id in self.vectors:
            self.vectors[identity_id] += stimulus

class EnhancedPersonalityTensorPhysics(PersonalityTensorPhysics):
    def __init__(self):
        super().__init__()
        self.context_gravity = 0.1

    def update(self) -> None:
        super().update()
        # simple gravity toward origin
        for k in self.positions:
            self.positions[k] -= self.context_gravity * self.positions[k]
