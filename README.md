# 2D Systolic Array AI Accelerator

## Project objective
Build a Python-based architectural simulator for a tiled 2D systolic-array accelerator used for matrix multiplication, a core operation in many neural-network workloads.

## Requirements
- Python 3.9+
- NumPy
- Pandas
- Matplotlib

Install:
```powershell
python -m pip install numpy pandas matplotlib
```

Run:
```powershell
python main.py
```

## Files
- `main.py`: runs the demo and experiments.
- `sequential.py`: reference matrix multiplication and MAC counting.
- `systolic_array.py`: tiled systolic-array functional model and ideal cycle estimator.
- `performance.py`: runs configurations and writes a CSV.
- `visualization.py`: generates performance plots.
- `results/`: generated CSV and PNG graphs.

## Architecture model and assumptions
The PE array has P x P processing elements. Each PE performs one multiply-accumulate (MAC) operation per cycle at the ideal peak rate. The model uses output-stationary tiles: each output tile is accumulated while the inputs are conceptually streamed through the array. A tile with m output rows, n output columns and reduction length K is estimated to take m+n+K-2 cycles, assuming non-empty dimensions and ideal pipelining. Output tiles are processed sequentially.

This is a high-level architectural model, not RTL, FPGA implementation, or a detailed cycle-accurate simulator. The matrix result is computed with NumPy; the cycle count, throughput and utilization are architectural estimates. Python wall-clock runtime is not hardware execution time.

## Metrics
- MAC count: M x N x K.
- Ideal peak throughput: P² MAC/cycle.
- Useful throughput: total MACs / estimated cycles.
- PE utilization: total MACs / (P² x estimated cycles) x 100.
- Cycle speedup: sequential baseline cycles / estimated systolic cycles. The baseline assumes one MAC per cycle.

## Suggested project extension
Add INT8/INT16 arithmetic, energy estimates, convolution mapping, or a Verilog implementation.
