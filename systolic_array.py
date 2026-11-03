import math
import numpy as np

def systolic_matmul(A, B, array_size):
    """
    Functional tiled systolic-array model.

    Each P x P PE array computes one output tile at a time using
    output-stationary accumulation. A tile with dimensions (m,n,k)
    requires m+n+k-2 ideal pipeline cycles (for non-empty dimensions).

    This is an architectural cycle model, not a cycle-accurate RTL simulator.
    The numerical result is computed with NumPy and checked against the
    reference implementation in main.py.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2-D matrices.")
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner matrix dimensions must match.")
    if array_size < 1:
        raise ValueError("array_size must be at least 1.")

    M, K = A.shape
    _, N = B.shape
    output = np.zeros((M, N), dtype=np.int64)

    total_cycles = 0
    total_macs = M * N * K
    tile_records = []

    for row_start in range(0, M, array_size):
        m = min(array_size, M - row_start)
        for col_start in range(0, N, array_size):
            n = min(array_size, N - col_start)
            # One output tile is mapped to the PE array.
            tile_cycles = m + n + K - 2 if min(m, n, K) > 0 else 0
            total_cycles += tile_cycles
            tile_records.append({
                "row_start": row_start,
                "col_start": col_start,
                "tile_rows": m,
                "tile_cols": n,
                "reduction_length": K,
                "cycles": tile_cycles
            })
            output[row_start:row_start+m, col_start:col_start+n] = (
                A[row_start:row_start+m, :] @ B[:, col_start:col_start+n]
            )

    peak_macs_per_cycle = array_size * array_size
    useful_macs_per_cycle = total_macs / total_cycles if total_cycles else 0.0
    utilization = (
        100.0 * total_macs / (peak_macs_per_cycle * total_cycles)
        if total_cycles else 0.0
    )

    metrics = {
        "array_size": array_size,
        "total_macs": total_macs,
        "total_cycles": total_cycles,
        "peak_macs_per_cycle": peak_macs_per_cycle,
        "useful_macs_per_cycle": useful_macs_per_cycle,
        "pe_utilization_percent": utilization,
        "tile_count": len(tile_records),
        "tile_records": tile_records
    }
    return output, metrics
