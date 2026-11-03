import pandas as pd
from sequential import sequential_mac_count
from systolic_array import systolic_matmul

def run_experiments(matrix_sizes=(16, 32, 64), array_sizes=(2, 4, 8), seed=7):
    rows = []
    for n in matrix_sizes:
        # Deterministic integer inputs make runs reproducible.
        import numpy as np
        rng = np.random.default_rng(seed + n)
        A = rng.integers(0, 10, size=(n, n), dtype=np.int64)
        B = rng.integers(0, 10, size=(n, n), dtype=np.int64)
        macs = sequential_mac_count(A, B)
        sequential_cycles = macs  # baseline: one MAC per cycle

        for p in array_sizes:
            C, metrics = systolic_matmul(A, B, p)
            if not np.array_equal(C, A @ B):
                raise AssertionError(f"Incorrect result for N={n}, array={p}")
            speedup = sequential_cycles / metrics["total_cycles"]
            rows.append({
                "matrix_size": n,
                "array_size": f"{p}x{p}",
                "processing_elements": p*p,
                "total_macs": macs,
                "sequential_cycles": sequential_cycles,
                "systolic_cycles": metrics["total_cycles"],
                "cycle_speedup": speedup,
                "useful_macs_per_cycle": metrics["useful_macs_per_cycle"],
                "pe_utilization_percent": metrics["pe_utilization_percent"],
                "tile_count": metrics["tile_count"],
                "correctness": "PASS"
            })
    return pd.DataFrame(rows)
