import matplotlib.pyplot as plt
import numpy as np

problems = [
    "01_1LDB", "02_1ITU", "03_2CGA", "04_5WN9", "05_5ZE9",
    "06_6E6R", "07_6E6R", "08_7AD5", "09_7CG5", "10_7WRK",
    "11_3TQB", "12_4JHW", "13_4JHW", "14_5IUS", "15_7A8S",
    "16_7BNY", "17_7DGW", "18_7MQQ", "19_7MQQ", "20_7UWL",
    "21_1B73", "22_1BCF", "23_1MPY", "24_1QY3", "25_2RKX",
    "26_3B5V", "27_4XOJ", "28_5YUI", "29_6CPA", "30_7UWL",
]

# RFdiffusion values (updated stats, score still 21.27)
rfdiff = [0, 3, 0, 5, 30, 42, 64, 0, 32, 1, 2, 0, 0, 1, 2, 2, 2, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0]

# Protpardelle-1c cc58 values (user-provided)
cc58 = [0, 1, 0, 4, 18, 47, 28, 0, 21, 6, 4, 0, 1, 2, 3, 2, 2, 0, 0, 0, 6, 1, 1, 5, 1, 1, 2, 6, 2, 0]

# Protpardelle-1c cc91-large values (from summary_by_problem.csv)
cc91 = [2, 3, 0, 5, 12, 63, 13, 0, 24, 4, 2, 2, 6, 2, 4, 2, 1, 1, 3, 0, 7, 1, 4, 6, 1, 1, 5, 10, 2, 0]

# MotifBench scores
rfdiff_score = 21.27
cc58_score = 28.16
cc91_score = 33.80

x = np.arange(len(problems))
width = 0.27

# Standard PowerPoint slide is 13.33 x 7.5 inches
fig, ax = plt.subplots(figsize=(13.33, 7.5))

# Match original plot colors
rfdiff_color = "#6BAED6"  # steel blue
cc58_color = "#74C476"    # green
cc91_color = "#E6842A"    # orange

bars1 = ax.bar(x - width, rfdiff, width, label=f"RFdiffusion: MotifBench score = {rfdiff_score}", color=rfdiff_color, edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x, cc58, width, label=f"Protpardelle-1c cc58: MotifBench score = {cc58_score}", color=cc58_color, edgecolor="white", linewidth=0.5)
bars3 = ax.bar(x + width, cc91, width, label=f"Protpardelle-1c cc91-large: MotifBench score = {cc91_score}", color=cc91_color, edgecolor="white", linewidth=0.5)

# Add value labels (only for non-zero values to reduce clutter)
for bars, vals, color in [(bars1, rfdiff, rfdiff_color), (bars2, cc58, cc58_color), (bars3, cc91, cc91_color)]:
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                str(val),
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=color,
            )

ax.set_xlabel("Problem", fontsize=14)
ax.set_ylabel("Number of Solutions", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(problems, rotation=60, ha="right", fontsize=10)
ax.tick_params(axis="y", labelsize=11)
ax.legend(loc="upper right", fontsize=12, framealpha=0.95, edgecolor="gray")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(max(rfdiff), max(cc58), max(cc91)) + 8)

plt.tight_layout()
plt.savefig(
    "/scratch/users/tianyulu/protpardelle-1c-sampling-results/03_motifbench_cc91-large/cc91_large_motifbench_results/mb-plot-3way.png",
    dpi=200,
    bbox_inches="tight",
)
print("Saved to mb-plot-3way.png")
