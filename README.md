# Land Cover Classification in Kazakh Steppes

**Sentinel-2 spectral indices + settlement proximity, with SHAP interpretability**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melomilk/kazakh-steppes/blob/master/KazakhSteppes.ipynb)

Code for the paper *"Land Cover Classification in Kazakh Steppes Using Sentinel-2 Spectral Indices and Settlement Proximity: A Machine Learning Approach"*, accepted to **Springer CCIS** (AIST 2026).

The project builds a land cover classifier for two dry steppe regions of Central Kazakhstan (**Betpak-Dala** and **Zhanaarka**), tests whether distance to the nearest settlement adds information beyond spectral indices, and uses SHAP to check which direction that feature pushes predictions.

---

## Key findings

1. **Settlement distance improves classification.** Adding `dist_settlement` to four spectral indices raises Random Forest accuracy on Betpak-Dala from **64.3% to 72.6%** (5-fold CV repeated 3×), and from 66.5% to 74.3% on a single 80/20 split.
2. **It matters most globally, less for bare land.** Across all classes, `dist_settlement` has the highest mean |SHAP|. For bare/degraded land (WorldCover class 60) it ranks 4th of 5; vegetation indices (NDVI, EVI) do most of the work.
3. **The piosphere pattern is reversed.** The grazing-pressure hypothesis predicts degraded land near settlements. The data shows the opposite: vegetation is greenest near settlements and bare land dominates farther away.

| Distance to settlement (Zhanaarka) | 0–1 km | 1–2 km | 2–5 km | 5–10 km | 10–20 km | 20 km+ |
|---|---|---|---|---|---|---|
| Mean NDVI | 0.326 | 0.344 | 0.286 | 0.229 | 0.216 | 0.129 |
| Mean BSI | −0.004 | 0.029 | 0.045 | 0.085 | 0.104 | 0.138 |

---

## Research question

> Does distance to the nearest human settlement carry information about land cover, and specifically about bare/degraded land, beyond what spectral indices capture?

The motivating hypothesis is the **piosphere effect**: in arid grazing lands, livestock concentrate around water points and settlements, so degradation should be highest close to them and decrease outward. A prior GIS study of Zhanaarka reported degradation within ~5 km of settlements.

This repo turns that hypothesis into an ML test:

- **Ablation:** if proximity is informative, a classifier with `dist_settlement` should outperform one without it.
- **Direction (SHAP):** if the piosphere effect holds, being *close* to a settlement should push predictions *toward* the bare/degraded class.

The first held. The second did not.

---

## Pipeline

```
Sentinel-2 L2A (Jun–Aug 2023, <20% cloud, median composite)
        │
        ├─► NDVI, BSI, NDWI, EVI ─────────────┐
                                              │
GHSL built-up surface (2020) ─► settlement    ├─► stack ─► stratified sample ─► CSV ─► RF / XGBoost / SVM ─► SHAP
     mask (>0.1) ─► distance transform ───────┤          (≤500 pts/class,
                                              │           100 m, seed 42)
ESA WorldCover v200 (2021) ─► labels ─────────┘
```

**Stage 1 — Google Earth Engine (JavaScript):** builds features and labels, draws stratified samples, exports CSVs to Google Drive.
**Stage 2 — Python notebook (Colab):** trains and evaluates the models, runs the ablation, SHAP, and distance-trend analysis.

### Study areas

| Region | Bounding box (lon/lat) | Role | Samples |
|---|---|---|---|
| Betpak-Dala | `[67, 44, 76, 50]` | Ablation (with vs. without `dist_settlement`), SHAP | 4,308 |
| Zhanaarka district (Ulytau) | `[69, 47, 74, 50]` | Cross-validated model comparison, distance-trend analysis | 4,001 (4,000 after dropping class 100) |

Note: the Zhanaarka box lies inside the Betpak-Dala box. See [Limitations](#limitations).

### Features

| Feature | Formula (Sentinel-2 bands) | Measures |
|---|---|---|
| NDVI | (B8 − B4) / (B8 + B4) | Vegetation greenness |
| EVI | 2.5 × (B8 − B4) / (B8 + 6·B4 − 7.5·B2 + 1) | Vegetation, soil/atmosphere-corrected (see [Known issues](#known-issues)) |
| BSI | ((B11 + B4) − (B8 + B2)) / ((B11 + B4) + (B8 + B2)) | Bare soil exposure |
| NDWI | (B3 − B8) / (B3 + B8) | Surface water / moisture |
| `dist_settlement` | Euclidean distance (m) to nearest GHSL pixel with `built_surface > 0.1` | Proximity to human settlement |

### Labels

[ESA WorldCover v200](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) (10 m, 2021). Nine classes occur in the study areas:

| Code | Class | Code | Class |
|---|---|---|---|
| 10 | Tree cover | 60 | **Bare / sparse vegetation** (degraded-land proxy) |
| 20 | Shrubland | 80 | Permanent water |
| 30 | Grassland | 90 | Herbaceous wetland |
| 40 | Cropland | 100 | Moss and lichen |
| 50 | Built-up | | |

After `LabelEncoder`, codes map to indices 0–8 in the order above (10→0, 20→1, …, 100→8). SHAP plots use these indices.

### Models

| Model | Settings | Preprocessing |
|---|---|---|
| Random Forest | `n_estimators=100, random_state=42` | Raw features |
| XGBoost | defaults, `eval_metric='mlogloss', random_state=42` | Raw features |
| SVM | RBF kernel, defaults, `random_state=42` | `StandardScaler` (required, see below) |

Tree models split on thresholds, so feature scale doesn't affect them. SVM uses distances between points, so without scaling `dist_settlement` (0–70,000 m) drowns out the indices (−1 to 1).

---

## Results

### Betpak-Dala: settlement-distance ablation (Random Forest)

| Features | Single 80/20 split | 5-fold CV × 3 repeats |
|---|---|---|
| 4 spectral indices | 66.47% | 64.33% ± 1.36% |
| 4 indices + `dist_settlement` | **74.25%** | **72.60% ± 1.08%** |

Per-class F1 changes (single split), largest gains in bold:

| Class | 10 | 20 | 30 | 40 | 50 | 60 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| 4 features | 0.74 | 0.55 | 0.54 | 0.74 | 0.61 | 0.70 | 0.96 | 0.49 | 0.63 |
| 5 features | 0.74 | 0.60 | 0.58 | 0.75 | **0.89** | 0.76 | 0.96 | **0.65** | **0.79** |

Most of the gain comes from built-up (50), wetland (90), and moss/lichen (100). Bare/degraded land (60) improves from 0.70 to 0.76.

### Betpak-Dala: model comparison (4 features, single split)

| Random Forest | XGBoost | SVM (unscaled) |
|---|---|---|
| 66.47% | 66.71% | 56.38% |

### Zhanaarka: model comparison (5 features, 5-fold CV)

| Model | Accuracy |
|---|---|
| Random Forest | 64.83% ± 1.37% |
| XGBoost | 64.38% ± 1.93% |
| SVM, unscaled | 28.48% ± 0.76% |
| SVM, standardized | 62.08% ± 1.94% |

RF and XGBoost are statistically indistinguishable. Unscaled SVM collapses, which is why preprocessing has to be reported when comparing classifiers.

### Feature importance (5-feature RF, Betpak-Dala)

| Rank | RF impurity importance | Mean \|SHAP\|, all classes | Mean \|SHAP\|, class 60 only |
|---|---|---|---|
| 1 | NDWI (0.227) | `dist_settlement` (0.051) | NDVI (0.067) |
| 2 | `dist_settlement` (0.224) | NDWI (0.049) | EVI (0.059) |
| 3 | NDVI (0.213) | EVI (0.043) | NDWI (0.046) |
| 4 | EVI (0.180) | NDVI (0.042) | `dist_settlement` (0.039) |
| 5 | BSI (0.155) | BSI (0.030) | BSI (0.036) |

These measure different things. Globally, `dist_settlement` separates human-related classes (built-up has a median distance of 0 m). For bare land specifically, vegetation indices decide; distance refines borderline cases.

### Mean distance to settlement by class

| Class | Betpak-Dala mean (m) | Zhanaarka mean (m) |
|---|---|---|
| 50 Built-up | 96 | 73 |
| 10 Tree cover | 1,613 | 1,045 |
| 40 Cropland | 3,866 | 3,846 |
| 60 **Bare/sparse** | **10,259** (largest) | **7,241** (largest) |

In both regions, bare/sparse land is on average the farthest class from settlements.

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
├── KazakhSteppes.ipynb       # Python: training, ablation, CV, tuning, SHAP, distance trends
├── full_pipeline.js          # GEE: full feature + label pipeline for both regions, CSV export
├── zhanaarka_extraction.js   # GEE: Zhanaarka-only version of the export (same logic)
└── zhanaarkalocation.js      # GEE: early exploration, NDVI map of a northern box [67,49,75,54]
```

`full_pipeline.js` is organized in steps:

| Step | What it does |
|---|---|
| 1 | NDVI visualization (exploration) |
| 2 | NDVI change 2019 vs 2023 (exploration, not used in the paper) |
| 3 | Compute NDVI, BSI, NDWI, EVI |
| 4 | Load ESA WorldCover labels |
| 5 | Build `dist_settlement` from GHSL |
| 6 | Betpak-Dala: stack, sample, export `betpak_samples_v2.csv` |
| 7 | Zhanaarka: same pipeline, export `zhanaarka_samples.csv` |

---

## Reproducing the results

### 1. Generate the data in Google Earth Engine

1. Sign up for [Google Earth Engine](https://earthengine.google.com/) (free for research).
2. Open the [Code Editor](https://code.earthengine.google.com/), paste `full_pipeline.js`, and click **Run**.
3. In the **Tasks** tab, run the `betpak_samples_v2` and `zhanaarka_samples` export tasks. The CSVs are saved to your Google Drive.

Stratified sampling uses `seed: 42`, so the same points come out each run as long as the source datasets are unchanged.

### 2. Run the notebook

1. Open the notebook with the **Open in Colab** badge above.
2. Upload `betpak_samples_v2.csv` and `zhanaarka_samples.csv` to Colab's `sample_data/` folder. The 4-feature baseline cell also reads `betpak_samples.csv`; `betpak_samples_v2.csv` contains the same points plus `dist_settlement`, so it can be used in its place.
3. Run all cells.

Dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn` (all available in Colab; `shap` is installed in the notebook).

### Exported CSV format

| Column | Description |
|---|---|
| `BSI`, `EVI`, `NDVI`, `NDWI` | Spectral indices at the sample point |
| `dist_settlement` | Distance to nearest settlement pixel, meters |
| `label` | WorldCover class code |
| `system:index`, `.geo` | GEE metadata, dropped in the notebook |

---

## Limitations

These matter for interpreting the results, and are good starting points for follow-up work.

- **Regions overlap.** Zhanaarka's box sits inside Betpak-Dala's, so Zhanaarka is a sub-region analysis rather than an independent validation site.
- **Random, not spatial, cross-validation.** Nearby points are similar, and `dist_settlement` is spatially smooth, so random folds likely inflate accuracy. Spatial block CV would give a more honest estimate.
- **Labels are model output.** WorldCover is itself a classifier (~75% global accuracy), which caps achievable accuracy and adds label noise.
- **Class 60 is a proxy, not a degradation map.** "Bare/sparse vegetation" includes natural desert. Degradation is a change over time; this study uses a single summer snapshot.
- **Built-up leakage.** Built-up pixels sit at near-zero distance from GHSL settlements by construction, so part of the accuracy gain comes from identifying class 50.
- **Balanced sampling.** ≤500 points per class does not reflect real class proportions; reported accuracy is for a balanced set, not a wall-to-wall map.
- **Temporal mismatch.** GHSL 2020, WorldCover 2021, Sentinel-2 2023. Likely minor in sparsely populated steppe.
- **Edge effects.** The settlement layer is clipped to each box before the distance transform, so settlements just outside the box are ignored.

## Known issues

- **EVI scaling.** `COPERNICUS/S2_SR_HARMONIZED` stores reflectance as integers scaled by 10,000. The EVI formula assumes reflectance in 0–1 (the `+1` term and the coefficients depend on it). Because the bands are not divided by 10,000 first, EVI values fall outside the valid range (e.g. ~2.4 for tree cover, ~−7 for water). The values still rank pixels consistently, so the classifier can use them, but they are not true EVI. Fix: `image.divide(10000)` before computing indices. NDVI, BSI, and NDWI are ratios and are unaffected.
- **Notebook cleanup.** The note in cell 2 about a `FileNotFoundError` is outdated; the Zhanaarka CV cell is duplicated; the XGBoost and SVM `GridSearchCV` cells have no saved outputs (tuned RF reached 72.73% CV accuracy).

---

## Ideas for future work

If you want to build on this:

- **Spatial cross-validation** (spatial blocks or leave-region-out) to measure true generalization.
- **Add a distance-to-water feature** (e.g. from JRC Global Surface Water) to separate the water effect from the settlement effect.
- **Time series instead of a snapshot.** NDVI trends across 2016–2024 would measure degradation as a process, which is what the piosphere hypothesis is about.
- **Finer distance bins near settlements** (0–500 m, 500 m–1 km) to check whether a piosphere ring exists at a scale 100 m sampling hides.
- **Exclude class 50** and rerun the ablation to measure the gain on non-built-up land only.
- **Independent region** outside the Betpak-Dala box, e.g. western or southern Kazakhstan.
- **More features:** elevation and slope (SRTM), precipitation (CHIRPS), soil type, distance to roads.

---

## Data sources

| Dataset | GEE ID | Use |
|---|---|---|
| Sentinel-2 Level-2A (surface reflectance) | `COPERNICUS/S2_SR_HARMONIZED` | Spectral indices |
| ESA WorldCover v200 | `ESA/WorldCover/v200` | Reference labels |
| JRC Global Human Settlement Layer, built-up surface | `JRC/GHSL/P2023A/GHS_BUILT_S/2020` | Settlement mask and distance |

All are free and open. Sentinel-2 and WorldCover data © ESA / Copernicus; GHSL © European Commission JRC.

## Citation

If you use this code or build on the findings, please cite:

```bibtex
@inproceedings{pak2026kazakhsteppes,
  title     = {Land Cover Classification in Kazakh Steppes Using Sentinel-2 Spectral Indices and Settlement Proximity: A Machine Learning Approach},
  author    = {Pak, Milana},
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
