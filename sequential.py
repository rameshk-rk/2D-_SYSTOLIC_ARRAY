import numpy as np

def sequential_matmul(A, B):
    """Reference matrix multiplication using NumPy."""
    return A @ B

def sequential_mac_count(A, B):
    """Number of multiply-accumulate operations for A(MxK) @ B(KxN)."""
    return A.shape[0] * A.shape[1] * B.shape[1]
