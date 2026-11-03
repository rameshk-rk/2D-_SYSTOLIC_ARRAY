from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make_plots(df, output_dir="results"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Cycles by PE-array size, one line per matrix size.
    plt.figure()
    for matrix_size, group in df.groupby("matrix_size"):
        group = group.sort_values("processing_elements")
        plt.plot(group["array_size"], group["systolic_cycles"],
                 marker="o", label=f"{matrix_size}x{matrix_size}")
    plt.xlabel("Systolic array configuration")
    plt.ylabel("Estimated accelerator cycles")
    plt.title("Systolic Array Cycle Comparison")
    plt.grid(True)
    plt.legend(title="Matrix")
    plt.tight_layout()
    plt.savefig(out / "execution_cycles.png", dpi=180)
    plt.close()

    # 2. Cycle speedup.
    plt.figure()
    for matrix_size, group in df.groupby("matrix_size"):
        group = group.sort_values("processing_elements")
        plt.plot(group["array_size"], group["cycle_speedup"],
                 marker="o", label=f"{matrix_size}x{matrix_size}")
    plt.xlabel("Systolic array configuration")
    plt.ylabel("Cycle-based speedup")
    plt.title("Ideal Cycle-Based Speedup vs Sequential Baseline")
    plt.grid(True)
    plt.legend(title="Matrix")
    plt.tight_layout()
    plt.savefig(out / "speedup.png", dpi=180)
    plt.close()

    # 3. PE utilization.
    plt.figure()
    for matrix_size, group in df.groupby("matrix_size"):
        group = group.sort_values("processing_elements")
        plt.plot(group["array_size"], group["pe_utilization_percent"],
                 marker="o", label=f"{matrix_size}x{matrix_size}")
    plt.xlabel("Systolic array configuration")
    plt.ylabel("PE utilization (%)")
    plt.title("Processing Element Utilization")
    plt.grid(True)
    plt.legend(title="Matrix")
    plt.tight_layout()
    plt.savefig(out / "utilization.png", dpi=180)
    plt.close()

    # 4. Throughput.
    plt.figure()
    for matrix_size, group in df.groupby("matrix_size"):
        group = group.sort_values("processing_elements")
        plt.plot(group["array_size"], group["useful_macs_per_cycle"],
                 marker="o", label=f"{matrix_size}x{matrix_size}")
    plt.xlabel("Systolic array configuration")
    plt.ylabel("Useful MAC operations per cycle")
    plt.title("Systolic Array Throughput")
    plt.grid(True)
    plt.legend(title="Matrix")
    plt.tight_layout()
    plt.savefig(out / "throughput.png", dpi=180)
    plt.close()


def make_architecture_diagram(output_path="architecture/systolic_array_4x4.png"):
    """Create a simple conceptual 4x4 PE-grid architecture diagram."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    n = 4
    for i in range(n):
        for j in range(n):
            x, y = j, n - 1 - i
            ax.add_patch(plt.Rectangle((x-0.35, y-0.25), 0.7, 0.5,
                                       fill=False, linewidth=1.6))
            ax.text(x, y, f"PE({i},{j})\\nMAC", ha="center", va="center", fontsize=9)
            if j < n-1:
                ax.annotate("", xy=(x+0.62, y), xytext=(x+0.38, y),
                            arrowprops=dict(arrowstyle="->", lw=1.2))
            if i < n-1:
                ax.annotate("", xy=(x, y-0.62), xytext=(x, y-0.38),
                            arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(-0.9, 1.5, "A rows →", rotation=90, ha="center", va="center")
    ax.text(1.5, 4.0, "B columns ↓", ha="center", va="center")
    ax.set_xlim(-1.2, 3.8)
    ax.set_ylim(-0.8, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Conceptual 4×4 Systolic Array (each PE performs MAC)")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
