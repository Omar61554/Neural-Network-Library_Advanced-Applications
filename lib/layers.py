"""Placeholder for layer implementations."""

class Dense:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # placeholder implementation
        return x


def example_layer():
    return Dense(10, 10)
