"""Placeholder activation functions."""

def relu(x):
    return max(0, x)

def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))
