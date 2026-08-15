import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess

file_path = "RefinedSource.xlsx"
time_col_start = "Cycle 1"
time_col_end   = "Cycle 40"
ce_col = "CE"

n_clusters = 3
test_size = 0.3

out_dir = "RTE_PhasePhysics"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_excel(file_path)

X = df.loc[:, time_col_start:time_col_end].values
CE = df[ce_col].values

X = StandardScaler().fit_transform(X)

rte = RandomTreesEmbedding(
    n_estimators=300,
    max_depth=6,
    random_state=0
)

X_rte = rte.fit_transform(X)
Z = PCA(n_components=2, random_state=0).fit_transform(X_rte.toarray())

Dim1 = Z[:,0]
Dim2 = Z[:,1]

km = KMeans(n_clusters=n_clusters, random_state=0)
labels_raw = km.fit_predict(Z)

def reorder_clusters_by_dim1(centers, labels):
    order = np.argsort(centers[:,0])
    new_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(order):
        new_labels[labels == old_id] = new_id + 1
    return new_labels, centers[order]

labels, centers = reorder_clusters_by_dim1(km.cluster_centers_, labels_raw)

def bootstrap_ci(data, n_boot=2000, ci=95):
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(sample.mean())
    low = np.percentile(means, (100-ci)/2)
    high = np.percentile(means, 100-(100-ci)/2)
    return low, high

def kernel_regression(x, y, bandwidth=0.2, x_grid=None):
    if x_grid is None:
        x_grid = np.linspace(x.min(), x.max(), 200)
    y_est = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        w = np.exp(-0.5*((x - x0)/bandwidth)**2)
        y_est[i] = np.sum(w*y) / np.sum(w)
    return x_grid, y_est

def compute_kde(x, y):
    kde = gaussian_kde(np.vstack([x, y]))
    return kde

def kde_grid(kde, x, y, n=200):
    xi = np.linspace(x.min(), x.max(), n)
    yi = np.linspace(y.min(), y.max(), n)
    Xg, Yg = np.meshgrid(xi, yi)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()]))
    return xi, yi, Z.reshape(Xg.shape)

def lowess_smooth(x, y, frac=0.3):
    order = np.argsort(x)
    smoothed = lowess(y[order], x[order], frac=frac, return_sorted=True)
    return smoothed[:,0], smoothed[:,1]

df_out = df.copy()
df_out["Dim1"] = Dim1
df_out["Dim2"] = Dim2
df_out["RTE_Cluster"] = labels

save_xlsx = os.path.join(out_dir, "RefinedResult.xlsx")
df_out.to_excel(save_xlsx, index=False)

print("\n========== RTE Embedding & Clustering Completed ==========")
print("Dim1, Dim2 computed from Random Trees Embedding + PCA")
print("Clusters reordered by Dim1 (physical evolution order)")
print("Saved file:", save_xlsx)
print("=========================================================\n")

def plot_phase_voronoi(Z, values, threshold, centers, title, cbar_label,
                        save_path):

    vor = Voronoi(centers)
    fig, ax = plt.subplots(figsize=(7,6))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False,
                    line_colors='k', line_width=1.1, point_size=0)

    mask_low  = values < threshold
    mask_high = values >= threshold

    ax.scatter(Z[mask_low,0], Z[mask_low,1],
               c='black', s=35, alpha=0.5)

    sc = ax.scatter(Z[mask_high,0], Z[mask_high,1],
                    c=values[mask_high], cmap='coolwarm',
                    s=45, alpha=0.8)

    ax.scatter(centers[:,0], centers[:,1],
               marker='*', s=260, c='gold',
               edgecolors='black', linewidths=1.2, zorder=10)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel("Dim-1 (RTE Evolution Coordinate)")
    ax.set_ylabel("Dim-2")

    dx = np.ptp(Z[:,0])
    dy = np.ptp(Z[:,1])
    ax.set_xlim(Z[:,0].min()-0.15*dx, Z[:,0].max()+0.15*dx)
    ax.set_ylim(Z[:,1].min()-0.15*dy, Z[:,1].max()+0.15*dy)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_kde_cloud(x, y, xlabel, ylabel, title, fname):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    plt.figure(figsize=(6,5))
    sc = plt.scatter(x, y, c=z, s=40, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="Kernel Density")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=600)
    plt.close()

def plot_kde_contour(x, y, xlabel, ylabel, title, fname):
    xi, yi, zi = kde_grid(gaussian_kde(np.vstack([x, y])), x, y)

    plt.figure(figsize=(6,5))
    plt.contourf(xi, yi, zi, levels=30, cmap='inferno')
    plt.colorbar(label="Probability Density")
    plt.scatter(x, y, s=10, c='white', alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=600)
    plt.close()

def plot_physics_regression(x, y, xlabel, ylabel, title, fname):

    xg, yk = kernel_regression(x, y, bandwidth=0.25)
    xl, yl = lowess_smooth(x, y, frac=0.3)

    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=25, alpha=0.4)
    plt.plot(xg, yk, c='red', lw=2, label="Kernel Regression")
    plt.plot(xl, yl, c='blue', lw=2, linestyle='--', label="LOWESS")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=600)
    plt.close()

plot_phase_voronoi(Z, CE, 94, centers,
                    "RTE Phase Map (Coulombic Efficiency)",
                    "CE (≥94)",
                    os.path.join(out_dir, "RTE_PhaseMap_CE.png"))

plot_kde_cloud(Dim1, CE, "Dim-1", "CE",
               "Dim-1 vs CE (KDE Cloud)",
               "Dim1_CE_KDE.png")

plot_kde_contour(Dim1, CE, "Dim-1", "CE",
                 "Dim-1 vs CE (Density Contour)",
                 "Dim1_CE_Contour.png")

plot_physics_regression(Dim1, CE, "Dim-1", "CE",
                         "Dim-1 → CE (Kernel + LOWESS)",
                         "Dim1_CE_PhysicsRegression.png")

print("All physics-informed phase diagrams and regression maps generated.")
print("Output directory:", out_dir)
