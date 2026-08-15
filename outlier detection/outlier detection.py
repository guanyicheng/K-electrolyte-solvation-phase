import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import gaussian_kde
from sklearn.linear_model import Ridge

# ============ Load data ============
file = "RTE_PhasePhysics_withEmbedding.xlsx"
df = pd.read_excel(file)

Dim1 = df["Dim1"].values
CE   = df["CE"].values

X = np.vstack([Dim1, CE])

# ============ 1. KDE density ============
kde = gaussian_kde(X)
density = kde(X)
df["KDE_density"] = density

threshold_kde = np.percentile(density, 10)
kde_outlier = density < threshold_kde

# ============ 2. Regression residual ============
model = Ridge(alpha=0.5)
model.fit(Dim1.reshape(-1, 1), CE)
CE_pred = model.predict(Dim1.reshape(-1, 1))
residual = CE - CE_pred
sigma = residual.std()

reg_outlier = np.abs(residual) > 2 * sigma
df["Residual"] = residual

# ============ 3. LOF local outlier factor ============
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.3)
lof_flag = lof.fit_predict(np.column_stack([Dim1, CE])) == -1

# ============ Consensus criterion ============
df["Outlier_KDE"] = kde_outlier
df["Outlier_Reg"] = reg_outlier
df["Outlier_LOF"] = lof_flag
df["Outlier_Final"] = (
    (kde_outlier & reg_outlier) |
    (kde_outlier & lof_flag)   |
    (reg_outlier & lof_flag)
)

# ============ Save results ============
df.to_excel("RTE_Outliers_Physical0.3.xlsx", index=False)

# ============ Visualization ============
xi, yi = np.mgrid[Dim1.min():Dim1.max():200j, CE.min():CE.max():200j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
zi = zi.reshape(xi.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xi, yi, zi, levels=30, cmap="inferno")
plt.scatter(Dim1, CE, c="white", s=10, alpha=0.4)

out = df["Outlier_Final"]
plt.scatter(Dim1[out], CE[out],
            s=80, facecolors='none', edgecolors='cyan', linewidths=2,
            label="Physical Outliers")

plt.xlabel("Dim-1 (Collective Solvation Coordinate)")
plt.ylabel("Coulombic Efficiency (%)")
plt.title("Free Energy Landscape with Physical Outliers")
plt.colorbar(label="Probability Density")
plt.legend()
plt.tight_layout()
plt.savefig("Dim1_CE_FreeEnergy_Outliers0.3.png", dpi=600)
plt.close()

print("Outlier detection completed.")
print(f"  Total outliers: {out.sum()}")
print("  Output: RTE_Outliers_Physical0.3.xlsx")
print("  Figure: Dim1_CE_FreeEnergy_Outliers0.3.png")
