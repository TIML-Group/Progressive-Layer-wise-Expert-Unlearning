import os

import matplotlib.pyplot as plt


neggrad_plus = [0.12, 0.9, 3.88, 11.98, 19.06, 22.86, 26.22, 28.14, 28.76, 28.9]
seuf_neggrad_plus = [0.84, 12.24, 25.6, 34.68, 41.66, 54.9, 60.58, 65.3, 67.44, 67.94]
seuf_ga = [11.28, 29.04, 39.22, 55.58, 62.12, 68.82, 75.5, 78.16, 79.52, 79.78]
ga = [33.52, 34.46, 35.52, 36.28, 36.84, 37.38, 37.68, 37.84, 37.86, 37.86]
mcu = [1.24, 2.54, 5.9, 11.82, 18.56, 24.3, 28.06, 30.16, 31.0, 31.2]
neus = [2.96, 4.66, 7.16, 11.04, 14.64, 18.1, 20.62, 22.58, 23.4, 23.6]


def main():
    steps = list(range(1, 11))
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
    })

    plt.figure(figsize=(9.2, 5.8))

    baseline_series = [
        ("NegGrad+", neggrad_plus, "#7f7f7f", "o"),
        ("NegGrad+ (SEUF)", seuf_neggrad_plus, "#c97b63", "s"),
        ("GA", ga, "#4e79a7", "^"),
        ("GA (SEUF)", seuf_ga, "#af7aa1", "D"),
        ("MCU", mcu, "#59a14f", "P"),
    ]

    for label, values, color, marker in baseline_series:
        plt.plot(
            steps,
            values,
            label=label,
            color=color,
            linestyle="--",
            linewidth=1.8,
            marker=marker,
            markersize=5,
            alpha=0.9,
        )

    plt.plot(
        steps,
        neus,
        label="PLEU (Ours)",
        color="#c62828",
        linestyle="-",
        linewidth=3.0,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.6,
        zorder=5,
    )

    plt.annotate(
        "PLEU (Ours)",
        xy=(steps[-1], neus[-1]),
        xytext=(7.25, 16.2),
        fontsize=12,
        color="#c62828",
        arrowprops={"arrowstyle": "->", "color": "#c62828", "lw": 1.2},
        zorder=2,
    )

    plt.xlabel("Relearning Step")
    plt.ylabel("Forget Accuracy on Train Set (%)")
    plt.title("Relearning Dynamics of Different Unlearning Methods")
    plt.xticks(steps)
    plt.yticks(range(0, 81, 10))
    plt.xlim(1, 10)
    plt.ylim(0, 80)
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.3)
    plt.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cccccc")
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relearning_fa_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
