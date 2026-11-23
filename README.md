# Land Cover Classification in Kazakh Steppes

**Sentinel-2 spectral indices + settlement proximity, with SHAP interpretability**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melomilk/kazakh-steppes/blob/master/KazakhSteppes_v3.ipynb)

Code for the paper *"Land Cover Classification in Kazakh Steppes Using Sentinel-2 Spectral Indices and Settlement Proximity: A Machine Learning Approach"* (M. Pak, A. Baubekova), accepted to **Springer CCIS** (AIST 2026).

The project builds a land cover classifier for two dry steppe regions of central Kazakhstan (**Betpak-Dala** and **Zhanaarka**), tests whether distance to the nearest settlement adds information beyond spectral indices, and uses SHAP to check which direction that feature pushes predictions.

> **Version note (Nov 2026).** The analysis was rerun after fixing EVI scaling and adding spatial cross-validation. All numbers below and in the published paper come from the v3 pipeline (`full_pipeline.js` + `KazakhSteppes_v3.ipynb`). The original notebook is kept in `archive/` for reference. See [Changelog](#changelog).

---

## Key findings

1. **Settlement distance improves classification, but less than random CV suggests.** Adding `dist_settlement` to four spectral indices raises Random Forest accuracy on Betpak-Dala from **65.5% to 72.8%** (repeated 5-fold CV). About half of that gain comes from the built-up class, which sits at ~0 m from settlements by construction. Without built-up samples and with **spatial block CV**, the gain is **+1.1 to +3.1 points**: smaller, but still positive.
2. **Random CV overstates accuracy by ~7 points.** With whole 0.5° areas held out for testing, accuracy drops from 72.8% to 65.6% (56.9% with 1° blocks).
3. **Distance ranks 3rd in importance.** Globally and for bare land (WorldCover class 60), NDVI matters most; `dist_settlement` is roughly tied with NDWI for 2nd–3rd. Its largest SHAP contribution is to the built-up class.
4. **The piosphere pattern is reversed.** The grazing-pressure hypothesis predicts degraded land near settlements. The data shows the opposite: vegetation is greenest near settlements, and bare land dominates farther away (Spearman ρ = 0.88 between distance and its SHAP value for class 60). A similar reversal has been reported in Mongolian steppe ([Sternberg 2012](https://doi.org/10.1007/s10745-012-9539-7)).

| Distance to settlement (Zhanaarka) | 0–1 km | 1–2 km | 2–5 km | 5–10 km | 10–20 km | 20 km+ |
|---|---|---|---|---|---|---|
| Mean NDVI | 0.326 | 0.344 | 0.286 | 0.229 | 0.216 | 0.129 |
| Mean BSI | −0.004 | 0.028 | 0.045 | 0.085 | 0.104 | 0.138 |

The trend is unchanged when built-up samples are removed.

---

## Research question

> Does distance to the nearest human settlement carry information about land cover, and specifically about bare/degraded land, beyond what spectral indices capture?

The motivating hypothesis is the **piosphere effect**: in arid grazing lands, livestock concentrate around water points and settlements, so degradation should be highest close to them and decrease outward.

This repo turns that hypothesis into an ML test:

- **Ablation:** if proximity is informative, a classifier with `dist_settlement` should outperform one without it.
- **Direction (SHAP):** if the piosphere effect holds, being *close* to a settlement should push predictions *toward* the bare/degraded class.

The first held (modestly, under strict evaluation). The second did not.

---

## Pipeline

```
Sentinel-2 L2A (Jun–Aug 2023, <20% cloud, median composite, reflectance / 10000)
        │
        ├─► NDVI, BSI, NDWI, EVI ─────────────┐
                                              │
GHSL built-up surface (2020) ─► settlement    ├─► stack ─► stratified sample ─► CSV ─► RF / XGBoost / SVM ─► SHAP
     mask (any built-up) ─► distance ─────────┤          (≤500 pts/class,           (random, leakage-free,
                                              │           100 m, seed 42,             spatial block CV)
ESA WorldCover v200 (2021) ─► labels ─────────┘           with coordinates)
```

**Stage 1 — Google Earth Engine (JavaScript):** builds features and labels, draws stratified samples, exports CSVs to Google Drive.
**Stage 2 — Python notebook (Colab):** trains and evaluates the models, runs the ablation, spatial CV, SHAP, distance-trend analysis, and produces every number and figure in the paper.

### Study areas

| Region | Bounding box (lon/lat) | Role | Samples |
|---|---|---|---|
| Betpak-Dala | `[67, 44, 76, 50]` (~450,000 km², includes part of Lake Balkhash) | Ablation, tuning, SHAP | 4,308 (9 classes) |
| Zhanaarka district (Ulytau) | `[69, 47, 74, 50]` | Model comparison (5-fold CV), distance-trend analysis | 4,000 (8 classes; class 100 dropped) |

The Zhanaarka box lies inside the Betpak-Dala box, so Zhanaarka is a sub-region analysis, not independent validation.

### Features

| Feature | Formula (Sentinel-2 bands, reflectance 0–1) | Measures |
|---|---|---|
| NDVI | (B8 − B4) / (B8 + B4) | Vegetation greenness |
| EVI | 2.5 × (B8 − B4) / (B8 + 6·B4 − 7.5·B2 + 1) | Vegetation, soil/atmosphere-corrected; clipped to [−1, 1] |
| BSI | ((B11 + B4) − (B8 + B2)) / ((B11 + B4) + (B8 + B2)) | Bare soil exposure |
| NDWI (McFeeters) | (B3 − B8) / (B3 + B8) | Open water / wet surfaces |
| `dist_settlement` | Euclidean distance (m) to nearest GHSL cell with any built-up surface (`built_surface > 0.1` m² per 100 m cell) | Proximity to human settlement |

### Labels

[ESA WorldCover v200](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) (10 m, 2021). Nine classes occur in the study areas:

| Code | Class | Code | Class |
|---|---|---|---|
| 10 | Tree cover | 60 | **Bare / sparse vegetation** (degraded-land proxy) |
| 20 | Shrubland | 80 | Permanent water |
| 30 | Grassland | 90 | Herbaceous wetland |
| 40 | Cropland | 100 | Moss and lichen |
| 50 | Built-up | | |

### Models

| Model | Settings | Preprocessing |
|---|---|---|
| Random Forest | `n_estimators=100, random_state=42` | Raw features |
| XGBoost | defaults, `eval_metric='mlogloss', random_state=42` | Raw features |
| SVM | RBF kernel, defaults, `random_state=42` | `StandardScaler` (required, see below) |

Tree models split on thresholds, so feature scale doesn't affect them. SVM uses distances between points, so without scaling `dist_settlement` (0–73,000 m) drowns out the indices (−1 to 1).

---

## Results

### Betpak-Dala: settlement-distance ablation (Random Forest)

| Evaluation protocol | 4 indices | + `dist_settlement` | Gain (pp) |
|---|---|---|---|
| Single 80/20 split, all classes | 67.98 | 73.32 | +5.3 |
| Repeated stratified CV (5×3), all classes | 65.51 ± 1.50 | 72.75 ± 1.30 | +7.2 |
| Repeated stratified CV, without built-up (50) | 67.53 | 71.22 | +3.7 |
| Spatial block CV (0.5°), all classes | 59.70 | 65.58 | +5.9 |
| Spatial block CV (0.5°), without built-up | 56.30 | 59.43 | +3.1 |
| Spatial block CV (0.25° / 1°), without built-up | 60.50 / 50.71 | 61.55 / 53.31 | +1.1 / +2.6 |

Spatial block CV groups samples into 0.5° × 0.5° grid cells (227 cells, roughly 40 × 55 km) and uses `StratifiedGroupKFold`, so the model is always tested on areas it never saw.

Per-class F1 (single split):

| Class | 10 | 20 | 30 | 40 | 50 | 60 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| 4 features | 0.76 | 0.55 | 0.55 | 0.72 | 0.72 | 0.71 | 0.94 | 0.49 | 0.67 |
| 5 features | 0.76 | 0.56 | 0.60 | 0.75 | **0.88** | 0.75 | 0.95 | **0.59** | **0.75** |

### Model comparison

| Setting | Random Forest | XGBoost | SVM (standardized) |
|---|---|---|---|
| Betpak-Dala, 4 features, single split | 67.98 | 66.24 | 56.15 |
| Betpak-Dala, 5 features, tuned (grid search, repeated CV) | **73.00** | 72.52 | 68.69 |
| Zhanaarka, 5 features, 5-fold CV | **66.68 ± 1.30** | 65.55 ± 0.96 | 64.28 ± 1.72 |

Best grid-search settings: RF `n_estimators=300, max_depth=20, min_samples_leaf=2`; XGBoost `n_estimators=300, max_depth=5, learning_rate=0.05`; SVM `C=10, kernel='rbf'`. Unscaled SVM on Zhanaarka collapses to 28.52% ± 0.79%.

### Feature importance (5-feature RF, Betpak-Dala)

| Rank | RF impurity importance | Mean \|SHAP\|, all classes | Mean \|SHAP\|, class 60 only |
|---|---|---|---|
| 1 | NDVI (0.238) | NDVI (0.060) | NDVI (0.105) |
| 2 | NDWI (0.236) | NDWI (0.051) | NDWI (0.051) |
| 3 | `dist_settlement` (0.222) | `dist_settlement` (0.051) | `dist_settlement` (0.041) |
| 4 | BSI (0.157) | BSI (0.033) | BSI (0.026) |
| 5 | EVI (0.147) | EVI (0.020) | EVI (0.013) |

### Mean distance to settlement by class

| Class | Betpak-Dala mean (m) | Zhanaarka mean (m) |
|---|---|---|
| 50 Built-up | 96 | 73 |
| 10 Tree cover | 1,613 | 1,045 |
| 40 Cropland | 3,866 | 3,846 |
| 60 **Bare/sparse** | **10,259** (largest) | **7,241** (largest) |

### Possible explanations for the reversed piosphere signal

- Irrigated household plots and small-scale cultivation green up land near villages.
- Distant common pastures receive less investment and management.
- Settlements were historically sited near water, so `dist_settlement` partly acts as distance-to-water.
- WorldCover class 60 includes naturally barren desert, which is more common far from settlements.

The current feature set cannot separate these effects.

---

## Repository structure

```
kazakh-steppes/
├── KazakhSteppes_v3.ipynb    # Python: all analyses and figures in the published paper (with outputs)
├── full_pipeline.js          # GEE: feature + label pipeline for both regions, CSV export (v3)
├── zhanaarkalocation.js      # GEE: early exploration, NDVI map of a northern box [67,49,75,54]
└── archive/
    └── KazakhSteppes_v2_original.ipynb   # original notebook (unscaled EVI, random CV only)
```

`full_pipeline.js` is organized in steps:

| Step | What it does |
|---|---|
| 1 | NDVI visualization (exploration) |
| 2 | NDVI change 2019 vs 2023 (exploration, not used in the paper) |
| 3 | Compute NDVI, BSI, NDWI, EVI |
| 4 | Load ESA WorldCover labels |
| 5 | Build `dist_settlement` from GHSL |
| 6 | Betpak-Dala: stack, sample, export `betpak_samples_v3.csv` |
| 7 | Zhanaarka: same pipeline, export `zhanaarka_samples_v3.csv` |

---

## Reproducing the results

### 1. Generate the data in Google Earth Engine

1. Sign up for [Google Earth Engine](https://earthengine.google.com/) (free for research).
2. Open the [Code Editor](https://code.earthengine.google.com/), paste `full_pipeline.js`, and click **Run**.
3. In the **Tasks** tab, run the `betpak_samples_v3` and `zhanaarka_samples_v3` export tasks. The CSVs are saved to your Google Drive.

Stratified sampling uses `seed: 42`, so the same points come out each run as long as the source datasets are unchanged.

### 2. Run the notebook

1. Open the notebook with the **Open in Colab** badge above.
2. Upload `betpak_samples_v3.csv` and `zhanaarka_samples_v3.csv` to the main Colab folder (not `sample_data/`).
3. Run all cells. The grid search in section 9 takes ~20–30 minutes on free Colab; it is much faster on a local multi-core machine.

Dependencies: `pandas`, `numpy`, `scikit-learn` (≥1.0, for `StratifiedGroupKFold`), `xgboost`, `shap`, `matplotlib`.

### Exported CSV format

| Column | Description |
|---|---|
| `BSI`, `EVI`, `NDVI`, `NDWI` | Spectral indices at the sample point (EVI is clipped to [−1, 1] in the notebook, not in the CSV) |
| `dist_settlement` | Distance to nearest settlement pixel, meters |
| `label` | WorldCover class code |
| `.geo` | Point geometry (GeoJSON); parsed into `lon`/`lat` in the notebook |
| `system:index` | GEE metadata, dropped in the notebook |

---

## Limitations

- **Regions overlap.** Zhanaarka's box sits inside Betpak-Dala's, so Zhanaarka is a sub-region analysis rather than an independent validation site.
- **Labels are model output.** WorldCover is itself a classifier, which caps achievable accuracy and adds label noise; accuracies measure agreement with WorldCover, not with field data. Moss/lichen (class 100) is particularly unreliable here.
- **Class 60 is a proxy, not a degradation map.** "Bare/sparse vegetation" includes natural desert. Degradation is a change over time; this study uses a single summer snapshot.
- **Balanced sampling.** ≤500 points per class does not reflect real class proportions; reported accuracy is for a balanced set, not a wall-to-wall map.
- **EVI over water.** Over bright water the EVI denominator approaches zero; 281 Betpak-Dala water samples had extreme values and were clipped to [−1, 1].
- **Temporal mismatch.** GHSL 2020, WorldCover 2021, Sentinel-2 2023. Likely minor in sparsely populated steppe.
- **Edge effects.** The settlement layer is clipped to each box before the distance transform, so settlements just outside the box are ignored.

---

## Ideas for future work

- **Hand-verified labels:** check a random, class-balanced subset of points against high-resolution imagery to measure WorldCover's error in the steppe and get a clean test set.
- **Add a distance-to-water feature** (e.g. JRC Global Surface Water) to separate the water effect from the settlement effect.
- **Time series instead of a snapshot.** NDVI trends across 2016–2025 would measure degradation as a process.
- **Finer distance bins near settlements** (0–500 m, 500 m–1 km) to check for a piosphere ring at a scale 100 m sampling may hide.
- **Independent region** outside the Betpak-Dala box, e.g. western or southern Kazakhstan.
- **More features:** elevation and slope (SRTM), precipitation (CHIRPS), soil type, distance to roads.
- **Geospatial foundation models** (e.g. Prithvi, Clay) compared against the tree-based baseline.

---

## Changelog

**v3 (Nov 2026), matches the published paper**
- EVI computed on reflectance scaled to 0–1 (`.divide(10000)`); previously computed on raw integer reflectance, which put EVI outside its valid range. NDVI, BSI and NDWI were unaffected.
- Sample points exported with coordinates (`geometries: true`). The sampled points are identical to v2.
- Added spatial block cross-validation, an ablation without the built-up class, grid search for all three models, and regenerated Figures 1–4.
- Accuracies changed by at most ~2 points; the model ranking and the reversed piosphere finding are unchanged. The settlement-distance gain and the SHAP ranking of `dist_settlement` are smaller than in v2.
- Removed `zhanaarka_extraction.js` (a duplicate of step 7 in `full_pipeline.js` with the old EVI bug).

**v2 (2026), submitted version.** See `archive/KazakhSteppes_v2_original.ipynb`.

---

## Data sources

| Dataset | GEE ID | Use |
|---|---|---|
| Sentinel-2 Level-2A (surface reflectance) | `COPERNICUS/S2_SR_HARMONIZED` | Spectral indices |
| ESA WorldCover v200 | `ESA/WorldCover/v200` | Reference labels |
| JRC Global Human Settlement Layer, built-up surface | `JRC/GHSL/P2023A/GHS_BUILT_S/2020` | Settlement mask and distance |

All are free and open. Contains modified Copernicus Sentinel data (2023). ESA WorldCover © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021), CC-BY 4.0. GHSL © European Commission JRC.

## Citation

If you use this code or build on the findings, please cite:

```bibtex
@inproceedings{pak2026kazakhsteppes,
  title     = {Land Cover Classification in Kazakh Steppes Using Sentinel-2 Spectral Indices and Settlement Proximity: A Machine Learning Approach},
  author    = {Pak, Milana and Baubekova, Assiya},
  booktitle = {Analysis of Images, Social Networks and Texts (AIST 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026}
}
```

## Contact

Milana Pak — Kazakh-British Technical University, Almaty
GitHub: [@melomilk](https://github.com/melomilk) · Email: pakmv2006@gmail.com

Issues and pull requests are welcome.
