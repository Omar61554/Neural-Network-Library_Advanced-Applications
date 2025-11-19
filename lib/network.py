"""High-level network orchestration placeholder."""

class Network:
    def __init__(self, layers=None):
        self.layers = layers or []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for l in self.layers:
            if hasattr(l, 'forward'):
                x = l.forward(x)
        return x
