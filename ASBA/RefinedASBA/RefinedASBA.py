"""
Adaptive Solvation Boundary Architecture (ASBA)
================================================

Modules:
  1. Adaptive Distribution Filter (ADF)
     - IQR-based boundary relaxation and outlier exclusion
  2. Phase Boundary Optimizer (PBO)
     - Dual-boundary search for LIA / MIA / HIA delineation

Metric:
  Cluster-Phase Agreement Score (CPAS) under three-phase
  (LIA / MIA / HIA) competition.

Outputs:
  - ASBA_Matrix_Landscape.tif / .png
  - PBO_Optimal_Landscape_CPAS_Normalized.tif / .png
  - ASBA_Optimized_IA_Result.xlsx
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable


# ==========================================================
# Global style
# ==========================================================

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 27
plt.rcParams['axes.unicode_minus'] = False


# ==========================================================
# Colormap and normalizations
# ==========================================================

bwr_nature = LinearSegmentedColormap.from_list(
    "bwr_nature",
    ["#4A90E2", "#F2E3DF", "#E35853"],
    N=256
)

matrix_norm = TwoSlopeNorm(vmin=0, vcenter=0.65, vmax=1.00)
pbo_norm = TwoSlopeNorm(vmin=0, vcenter=0.85, vmax=1.00)


# ==========================================================
# 1. Load data
# ==========================================================

file_path = "RefinedASBA.xlsx"

df = pd.read_excel(file_path)
S_raw = df["Associate"].values.astype(float)
labels_raw = df["Cluster"].values


# ==========================================================
# 2. Parameters
# ==========================================================

target_cluster = 3

lower_relaxation_vals = np.round(np.arange(0, 2.01, 0.2), 1)
upper_relaxation_vals = np.round(np.arange(0, 2.01, 0.2), 1)

boundary_candidate_num = 180
MIN_SAMPLE = 190
LOWER_PHASE_BOUNDARY_MIN = 0.15


# ==========================================================
# 3. Initialization
# ==========================================================

S_series = pd.Series(S_raw)
matrix_results = {}
global_score_min = 999
global_score_max = -999
global_best = {"score": -1}


# ==========================================================
# 4. ADF + PBO search
# ==========================================================

total = len(lower_relaxation_vals) * len(upper_relaxation_vals)
current = 0

print(f"Starting calculation: {total} subplots")

for upper_relaxation in upper_relaxation_vals:
    for lower_relaxation in lower_relaxation_vals:

        current += 1
        print(
            f"\rProcessing {current}/{total} | "
            f"lower={lower_relaxation}, upper={upper_relaxation}",
            end=""
        )

        q1 = S_series.quantile(0.25)
        q3 = S_series.quantile(0.75)
        iqr = q3 - q1

        lower_envelope = q1 - lower_relaxation * iqr
        upper_envelope = q3 + upper_relaxation * iqr
        envelope_mask = (S_raw >= lower_envelope) & (S_raw <= upper_envelope)

        S_clean = S_raw[envelope_mask]
        labels_clean = labels_raw[envelope_mask]

        if len(S_clean) < MIN_SAMPLE:
            matrix_results[(lower_relaxation, upper_relaxation)] = {
                "valid": False
            }
            continue

        lower_phase_boundaries = np.linspace(
            S_clean.min(), S_clean.max(), boundary_candidate_num
        )
        upper_phase_boundaries = np.linspace(
            S_clean.min(), S_clean.max(), boundary_candidate_num
        )

        agreement_landscape = np.full(
            (boundary_candidate_num, boundary_candidate_num), np.nan
        )

        valid_exist = False
        best_score = -1
        best_idx = None

        for i, lower_phase_boundary in enumerate(lower_phase_boundaries):

            if lower_phase_boundary <= LOWER_PHASE_BOUNDARY_MIN:
                continue

            for j, upper_phase_boundary in enumerate(upper_phase_boundaries):

                if upper_phase_boundary <= lower_phase_boundary:
                    continue

                ia_phases = np.where(
                    S_clean < lower_phase_boundary,
                    "LIA",
                    np.where(S_clean < upper_phase_boundary, "MIA", "HIA")
                )

                unique, counts = np.unique(ia_phases, return_counts=True)
                ratio_dict = dict(zip(unique, counts / len(ia_phases)))

                lia = ratio_dict.get("LIA", 0)
                mia = ratio_dict.get("MIA", 0)
                hia = ratio_dict.get("HIA", 0)

                if not (0.20 <= lia <= 0.45 and
                        0.20 <= mia <= 0.45 and
                        0.20 <= hia <= 0.45):
                    continue

                cluster_target_mask = (labels_clean == target_cluster)
                if not np.any(cluster_target_mask):
                    continue

                lia_mask = (ia_phases == "LIA")
                cpas_lia = (
                    0.8 * np.mean(ia_phases[cluster_target_mask] == "LIA")
                    + 0.2 * np.mean(labels_clean[lia_mask] == target_cluster)
                )

                mia_mask = (ia_phases == "MIA")
                cpas_mia = (
                    0.8 * np.mean(ia_phases[cluster_target_mask] == "MIA")
                    + 0.2 * np.mean(labels_clean[mia_mask] == target_cluster)
                )

                hia_mask = (ia_phases == "HIA")
                cpas_hia = (
                    0.8 * np.mean(ia_phases[cluster_target_mask] == "HIA")
                    + 0.2 * np.mean(labels_clean[hia_mask] == target_cluster)
                )

                phase_scores = {
                    "LIA": cpas_lia,
                    "MIA": cpas_mia,
                    "HIA": cpas_hia
                }
                winning_phase = max(phase_scores, key=phase_scores.get)
                cpas = phase_scores[winning_phase]

                agreement_landscape[j, i] = cpas
                valid_exist = True

                if cpas < global_score_min:
                    global_score_min = cpas
                if cpas > global_score_max:
                    global_score_max = cpas

                if cpas > best_score:
                    best_score = cpas
                    best_idx = (i, j)

                if cpas > global_best["score"]:
                    global_best = {
                        "score": cpas,
                        "winning_phase": winning_phase,
                        "cpas_lia": cpas_lia,
                        "cpas_mia": cpas_mia,
                        "cpas_hia": cpas_hia,
                        "lower_relaxation": lower_relaxation,
                        "upper_relaxation": upper_relaxation,
                        "lower_phase_boundary": lower_phase_boundary,
                        "upper_phase_boundary": upper_phase_boundary,
                        "agreement_landscape": agreement_landscape,
                        "lower_phase_boundaries": lower_phase_boundaries,
                        "upper_phase_boundaries": upper_phase_boundaries,
                        "best_idx": (i, j)
                    }

        if not valid_exist:
            matrix_results[(lower_relaxation, upper_relaxation)] = {
                "valid": False
            }
            continue

        valid_mask = ~np.isnan(agreement_landscape)
        ys, xs = np.where(valid_mask)

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        pad = 2
        xmin = max(0, xmin - pad)
        xmax = min(boundary_candidate_num - 1, xmax + pad)
        ymin = max(0, ymin - pad)
        ymax = min(boundary_candidate_num - 1, ymax + pad)

        cropped_landscape = agreement_landscape[ymin:ymax + 1, xmin:xmax + 1]

        matrix_results[(lower_relaxation, upper_relaxation)] = {
            "valid": True,
            "cropped_landscape": cropped_landscape,
            "best_x": best_idx[0] - xmin,
            "best_y": best_idx[1] - ymin,
            "best_score": best_score
        }

print("\n\nCalculation complete")


# ==========================================================
# 5. Global min-max normalization
# ==========================================================

for result in matrix_results.values():
    if result["valid"]:
        result["cropped_landscape"] = (
            (result["cropped_landscape"] - global_score_min)
            / (global_score_max - global_score_min)
        )

global_best["agreement_landscape"] = (
    (global_best["agreement_landscape"] - global_score_min)
    / (global_score_max - global_score_min)
)


# ==========================================================
# 6. ASBA matrix landscape
# ==========================================================

print("Plotting ASBA matrix landscape...")

fig, axes = plt.subplots(
    len(upper_relaxation_vals),
    len(lower_relaxation_vals),
    figsize=(17, 15),
    dpi=1200
)
plt.subplots_adjust(wspace=0.02, hspace=0.02)

for upper_idx, upper_relaxation in enumerate(upper_relaxation_vals):
    for lower_idx, lower_relaxation in enumerate(lower_relaxation_vals):

        ax = axes[upper_idx, lower_idx]
        result = matrix_results[(lower_relaxation, upper_relaxation)]

        if not result["valid"]:
            ax.set_facecolor("#BFBFBF")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        ax.imshow(
            result["cropped_landscape"],
            origin="lower",
            aspect="auto",
            cmap=bwr_nature,
            norm=matrix_norm
        )

        ax.scatter(
            result["best_x"],
            result["best_y"],
            s=18,
            c="red",
            edgecolors="white",
            linewidths=0.4
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

for lower_idx, lower_relaxation in enumerate(lower_relaxation_vals):
    axes[-1, lower_idx].set_xlabel(f"{lower_relaxation}")

for upper_idx, upper_relaxation in enumerate(upper_relaxation_vals):
    axes[upper_idx, 0].set_ylabel(f"{upper_relaxation}")

cbar_ax = fig.add_axes([0.92, 0.12, 0.025, 0.76])
cbar = fig.colorbar(
    ScalarMappable(norm=matrix_norm, cmap=bwr_nature),
    cax=cbar_ax
)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["0", "1"])
cbar.ax.tick_params(labelsize=27, length=6)

plt.savefig("ASBA_Matrix_Landscape.tif", dpi=1200, bbox_inches="tight")
plt.savefig("ASBA_Matrix_Landscape.png", dpi=1200, bbox_inches="tight")
plt.close(fig)

print("ASBA matrix landscape saved (TIFF + PNG)")


# ==========================================================
# 7. PBO optimal landscape
# ==========================================================

print("Plotting global optimal landscape...")

valid_mask = ~np.isnan(global_best["agreement_landscape"])
ys, xs = np.where(valid_mask)

xmin, xmax = xs.min(), xs.max()
ymin, ymax = ys.min(), ys.max()

pad = 3
xmin = max(0, xmin - pad)
xmax = min(boundary_candidate_num - 1, xmax + pad)
ymin = max(0, ymin - pad)
ymax = min(boundary_candidate_num - 1, ymax + pad)

cropped_best = global_best["agreement_landscape"][
    ymin:ymax + 1, xmin:xmax + 1
]
best_x = global_best["best_idx"][0] - xmin
best_y = global_best["best_idx"][1] - ymin

fig2, ax2 = plt.subplots(figsize=(7.84, 5), dpi=1200)

im2 = ax2.imshow(
    cropped_best,
    origin="lower",
    aspect="auto",
    cmap=bwr_nature,
    norm=pbo_norm
)

ax2.scatter(
    best_x,
    best_y,
    s=100,
    c="red",
    edgecolors="white",
    linewidth=1,
    zorder=10
)

xticks = np.linspace(0, cropped_best.shape[1] - 1, 5).astype(int)
yticks = np.linspace(0, cropped_best.shape[0] - 1, 5).astype(int)

xlabels = [
    f"{global_best['lower_phase_boundaries'][xmin + i]:.3f}"
    for i in xticks
]
ylabels = [
    f"{global_best['upper_phase_boundaries'][ymin + i]:.3f}"
    for i in yticks
]

ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_xticklabels(xlabels)
ax2.set_yticklabels(ylabels)

ax2.set_xlabel("Lower phase boundary", fontsize=20, labelpad=10)
ax2.set_ylabel("Upper phase boundary", fontsize=20, labelpad=10)

cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Normalized CPAS", fontsize=20, labelpad=2)
cbar2.ax.tick_params(labelsize=20)
cbar2.set_ticks([0, 1])
cbar2.set_ticklabels(["0", "1"])

for spine in ax2.spines.values():
    spine.set_linewidth(1)
ax2.tick_params(axis="both", width=1, length=4,labelsize=20)

plt.tight_layout()
plt.savefig(
    "PBO_Optimal_Landscape_CPAS_Normalized.tif",
    dpi=1200,
    bbox_inches="tight"
)
plt.savefig(
    "PBO_Optimal_Landscape_CPAS_Normalized.png",
    dpi=1200,
    bbox_inches="tight"
)
plt.close(fig2)

print("Global optimal landscape saved (TIFF + PNG)")


# ==========================================================
# 8. Export optimal classification
# ==========================================================

print("Exporting optimal classification...")

best_lower_relaxation = global_best["lower_relaxation"]
best_upper_relaxation = global_best["upper_relaxation"]
best_lower_boundary = global_best["lower_phase_boundary"]
best_upper_boundary = global_best["upper_phase_boundary"]

q1 = S_series.quantile(0.25)
q3 = S_series.quantile(0.75)
iqr = q3 - q1

lower_envelope = q1 - best_lower_relaxation * iqr
upper_envelope = q3 + best_upper_relaxation * iqr
envelope_mask = (S_raw >= lower_envelope) & (S_raw <= upper_envelope)

output_df = df[envelope_mask].copy().reset_index(drop=True)
S_clean = S_raw[envelope_mask]

output_df["IA_Category"] = np.where(
    S_clean < best_lower_boundary,
    "LIA",
    np.where(S_clean < best_upper_boundary, "MIA", "HIA")
)

out_excel = "ASBA_Optimized_IA_Result.xlsx"
output_df.to_excel(out_excel, index=False)

print(f"Saved: {out_excel}")
print(
    f"Optimal ADF | lower={best_lower_relaxation}, "
    f"upper={best_upper_relaxation}"
)
print(
    f"Optimal PBO | lower boundary={best_lower_boundary:.4f}, "
    f"upper boundary={best_upper_boundary:.4f}"
)

print("\n" + "=" * 60)
print("ASBA THREE-PHASE ION-ASSOCIATION COMPETITION RESULT")
print("=" * 60)
print(f"Winning Phase = {global_best['winning_phase']}")
print()
print(f"CPAS(LIA) = {global_best['cpas_lia']:.4f}")
print(f"CPAS(MIA) = {global_best['cpas_mia']:.4f}")
print(f"CPAS(HIA) = {global_best['cpas_hia']:.4f}")

print("\n" + "=" * 60)
print("IA PHASE DISTRIBUTION (optimal ADF + PBO)")
print("=" * 60)

phase_order = ["LIA", "MIA", "HIA"]
header = (
    f"{'Group':<14}"
    + "".join(f"{p:>10}" for p in phase_order)
    + f"{'n':>8}"
)
print(header)
print("-" * len(header))

n_total = len(output_df)
total_counts = output_df["IA_Category"].value_counts()
row = f"{'All samples':<14}"
for phase in phase_order:
    row += f"{total_counts.get(phase, 0) / n_total * 100:>9.2f}%"
row += f"{n_total:>8d}"
print(row)

for cluster_id in sorted(output_df["Cluster"].unique()):
    cluster_df = output_df[output_df["Cluster"] == cluster_id]
    n_cluster = len(cluster_df)
    counts = cluster_df["IA_Category"].value_counts()
    row = f"{'Cluster ' + str(cluster_id):<14}"
    for phase in phase_order:
        row += f"{counts.get(phase, 0) / n_cluster * 100:>9.2f}%"
    row += f"{n_cluster:>8d}"
    print(row)
