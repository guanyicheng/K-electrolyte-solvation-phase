# Code and data for: Data-driven discovery of an optimal solvation phase for potassium battery electrolytes

This repository contains the custom Python code and source data used in the
manuscript *"Data-driven discovery of an optimal solvation phase for potassium
battery electrolytes"* (Nature Communications, submission).

The pipeline (1) removes physically anomalous Coulombic-efficiency trajectories,
(2) embeds and clusters the remaining trajectories with Random Trees Embedding
+ K-means to reveal three global solvation tiers, (3) re-clusters the
top-performing tier to expose finer sub-structure, and (4–5) uses the Adaptive
Solvation Boundary Architecture (ASBA) to locate the optimal LIA/MIA/HIA
ion-association boundaries in both the global and refined analyses.

## Repository contents

```
.
├── README.md
├── LICENSE                         # MIT License (code)
├── requirements.txt
├── outlier detection/
│   ├── outlier detection.py
│   ├── RTE_PhasePhysics_withEmbedding.xlsx   # 774 CE trajectories + Raman IA data
│   └── RTE_Outliers_Physical0.3.xlsx         # output: outlier flags
├── Cluster/
│   ├── GlobalClustering/
│   │   ├── GlobalClustering.py
│   │   ├── GlobalSource.xlsx                 # 716 trajectories (outliers removed)
│   │   └── RTE_PhasePhysics/GlobalResult.xlsx
│   └── RefinedClustering/
│       ├── RefinedClustering.py
│       ├── RefinedSource.xlsx                # 255 trajectories (global tier 3)
│       └── RTE_PhasePhysics/RefinedResult.xlsx
└── ASBA/
    ├── GlobalASBA/
    │   ├── GlobalASBA.py
    │   ├── GlobalASBA.xlsx                   # GlobalResult + Associate + Cluster
    │   └── ASBA_Optimized_IA_Result.xlsx
    └── RefinedASBA/
        ├── RefinedASBA.py
        ├── RefinedASBA.xlsx
        └── ASBA_Optimized_IA_Result.xlsx
```

All intermediate `.xlsx` files are included, so each step can also be run on its
own. The data-preparation steps between scripts are simple spreadsheet filters
on columns produced by the preceding step:

1. **Outlier detection → Global source**: keep rows with `Outlier_Final = False`
   (774 → 716 samples) to obtain `GlobalSource.xlsx`.
2. **Global clustering → Refined source**: take rows with `RTE_Cluster = 3`
   (255 samples) from `GlobalResult.xlsx` to obtain `RefinedSource.xlsx`.
3. **Clustering → ASBA input**: carry the Raman-derived `Associate`
   (ion-association index) column and rename `RTE_Cluster` to `Cluster` to
   obtain `GlobalASBA.xlsx` / `RefinedASBA.xlsx`.

Every resulting file is versioned in this archive.

## Requirements

- **Operating system:** tested on Windows 10/11; the code is pure Python and
  also runs on Linux/macOS.
- **Python:** ≥ 3.9.
- **Packages:** numpy, pandas, scikit-learn, scipy, matplotlib, statsmodels,
  openpyxl (see `requirements.txt`).
- **Hardware:** standard laptop CPU; no GPU or special hardware required.

## Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

Typical install time: under 5 minutes on a current computer.

## Running the pipeline

Run the scripts **in order**, each from its own directory (the scripts read
their input `.xlsx` from the current working directory):

```bash
# 1. Outlier detection
cd "outlier detection"
python "outlier detection.py"
cd ..

# 2. Global clustering
cd "Cluster/GlobalClustering"
python GlobalClustering.py
cd ../..

# 3. Refined clustering
cd "Cluster/RefinedClustering"
python RefinedClustering.py
cd ../..

# 4. Global ASBA
cd "ASBA/GlobalASBA"
python GlobalASBA.py
cd ../..

# 5. Refined ASBA
cd "ASBA/RefinedASBA"
python RefinedASBA.py
cd ../..
```

Each script writes its result tables (`.xlsx`) and figures (`.png`/`.tif`) into
its own directory. The ASBA grid search is the longest step; total run time for
the full pipeline is a few minutes on a standard laptop.

## Key parameters (all as reported in the manuscript)

- Outlier detection: KDE 10th-percentile threshold; Ridge regression
  (`alpha = 0.5`) with a 2σ residual cut; `LocalOutlierFactor(n_neighbors = 20,
  contamination = 0.3)`; consensus = majority vote of the three detectors.
- Clustering: `StandardScaler` → `RandomTreesEmbedding(n_estimators = 300,
  max_depth = 6, random_state = 0)` → `PCA(n_components = 2)` →
  `KMeans(n_clusters = 3, random_state = 0)`; clusters reordered by Dim-1.
- ASBA: ADF relaxation grid 0–2.0 (step 0.2) × IQR; PBO boundary grid of 180
  candidates; three-phase ratio window 0.20–0.45 each; `MIN_SAMPLE = 630`
  (global) / `190` (refined); CPAS = 0.8 × phase-purity + 0.2 × cluster-purity.

## Reproducibility note

All hyperparameters and the source dataset used to construct the analysis are
exactly as reported in the manuscript. The code was cleaned and reorganised for
public release, and third-party library versions (notably scikit-learn, NumPy,
and SciPy) have advanced since the original calculations. Re-running the full
sequential pipeline may therefore produce minor numerical differences in the
embedded coordinates, cluster boundaries, and CPAS values relative to the exact
values shown in the published figures. The principal conclusions are
unaffected: the three-tier global solvation structure, the identification of
MIA as the optimal ion-association regime, and its external validation are all
reproduced. The `.xlsx` files and figures included in this archive are the
version used for the manuscript.

## License

- **Code:** released under the MIT License — see [LICENSE](LICENSE).
- **Data:** released under the Creative Commons Attribution 4.0 International
  License ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

## Contact

For questions about the code or data, please contact the corresponding author.
