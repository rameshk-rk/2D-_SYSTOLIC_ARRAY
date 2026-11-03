from pathlib import Path
import numpy as np
import pandas as pd
from sequential import sequential_matmul
from systolic_array import systolic_matmul
from performance import run_experiments
from visualization import make_plots, make_architecture_diagram

def main():
    Path("results").mkdir(exist_ok=True)
    Path("architecture").mkdir(exist_ok=True)

    print("=" * 65)
    print("  2D SYSTOLIC ARRAY AI ACCELERATOR - SIMULATION")
    print("=" * 65)

    # Small demonstration with readable matrices.
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [2, 3, 4, 5],
                  [1, 3, 5, 7]], dtype=np.int64)
    B = np.array([[1, 0, 2, 1],
                  [0, 1, 1, 2],
                  [2, 1, 0, 1],
                  [1, 2, 1, 0]], dtype=np.int64)

    print("\nInput Matrix A:")
    print(A)
    print("\nInput Matrix B:")
    print(B)

    reference = sequential_matmul(A, B)
    result, metrics = systolic_matmul(A, B, array_size=4)

    print("\nOutput Matrix C = A x B:")
    print(result)
    print("\nCorrectness check:", "PASS" if np.array_equal(reference, result) else "FAIL")
    print("\n4x4 array demonstration metrics:")
    for key in ("total_macs", "total_cycles", "peak_macs_per_cycle",
                "useful_macs_per_cycle", "pe_utilization_percent", "tile_count"):
        print(f"{key}: {metrics[key]}")

    print("\nRunning experiments for 16x16, 32x32 and 64x64 matrices...")
    df = run_experiments()
    df.to_csv("results/performance.csv", index=False)
    make_plots(df)
    make_architecture_diagram()

    print("\nExperiment results:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSaved files:")
    print("  results/performance.csv")
    print("  results/execution_cycles.png")
    print("  results/speedup.png")
    print("  results/utilization.png")
    print("  results/throughput.png")
    print("  architecture/systolic_array_4x4.png")
    print("\nSimulation completed successfully.")

if __name__ == "__main__":
    main()
