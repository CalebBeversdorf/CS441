# Timeguessr AI — Strategies & Design Decisions

A concise summary of the data, discretization, modeling, training, and evaluation strategies explored for **Timeguessr AI** (predicting a photo's capture year `1900–2026` and geographic coordinates `(lat, lon)` scored out of `10,000` points).

---

## 1. Data Curation & Leak-Free Splitting
- **Training/Validation Pool (`170,945` external images across `240` countries):** Combined 5 complementary geotagged + timestamped sources to cover both global geography and historical eras:
  - `OSV-5M` (`46,461`, global street-level, modern 2010s–2020s)
  - `YFCC15M` (`46,633`, global Flickr photos, ~2000–2014) + `YFCC-Pre2000` (`12,851`, pre-2000)
  - `PastVu` (`50,000`, historical 1900–1999, heavily skewed toward Russia) + `PastVu-NonRU` (`15,000`, targeted non-Russian historical photos to counter regional skew)
- **Zero-Leakage Evaluation:** Split external sources into `153,953` train / `16,992` val, and kept **`1,600` Timeguessr daily challenge photos** strictly held out for final testing (never used for tiling, training, or early stopping).
- **Sublinear Country × Decade Balanced Sampling:** Raw data is 30.3% Russia (`PastVu`) and heavily clustered in recent decades. We use a `WeightedRandomSampler` with sublinear weights $w_i \propto n_{\text{country}}^{\alpha_c - 1} \cdot n_{\text{decade}}^{\alpha_d - 1}$ (tuned from $\alpha_c = 0.6 \to 0.7$, $\alpha_d = 0.75$) to curb Russia dominance without over-boosting small European countries.

---

## 2. World Discretization (`03_s2_world_tiling.ipynb`)
Direct `(lat, lon)` regression collapses toward ocean/continental averages under visual ambiguity, so we frame geolocation as classification over discrete tiles. We compared two families on the held-out Timeguessr oracle ceiling:

| Partition | Tiles | Country Purity | Oracle Med Dist | Oracle Mean Loc Score (/5k) |
| :--- | :---: | :---: | :---: | :---: |
| `s2_coarse` (Adaptive S2 Quadtree) | 628 | 56.1% | 21.9 km | 4,737.1 |
| `s2_medium` (Data Spherical Centroid) | 1,404 | 74.1% | 9.0 km | 4,834.8 |
| `s2_medium` (Geometric S2 Center — ablation) | 1,404 | 74.1% | 34.7 km | 4,622.1 |
| `s2_fine` (Adaptive S2 Quadtree) | 2,688 | 83.5% | 3.9 km | 4,864.1 |
| **`semantic_medium` (Country-Aware Spherical K-Means)** | **1,249** | **100.0%** | **8.5 km** | **4,876.3** |

**Key findings:**
- Replacing S2 geometric cell centers with **data-weighted 3D spherical centroids** cut median oracle error by nearly 4× (`34.7 km → 9.0 km`) and gained `+213 pts`.
- **`semantic_medium`** (clustering *within* each country via Spherical K-Means) achieved the **highest oracle ceiling (`4,876.3`)** with fewer tiles than `s2_medium`, aligned tiles with cities/borders, and guaranteed **100% single-country purity** for downstream OCR language masking.

---

## 3. Dual-Head ViT Modeling & Loss Design (`src/timeguessr_ai/`)
- **Backbone Comparison:**
  - **`geolocal/StreetCLIP`** (`ViT-L/14@336px`, ~304M vision params) — CLIP pretrained specifically on geotagged street-level imagery.
  - **`google/siglip2-base-patch16-384`** (`ViT-B/16@384px`, ~86M vision params) — modern general-purpose vision-language encoder with higher input resolution and MAP attention pooling.
- **Dual MLP Heads:** Shared vision embedding feeds a **Time Head** (`127` classes for `1900–2026`) and a **Space Head** (`1,249` `semantic_medium` tiles).
- **Soft-Label Losses Aligned with Timeguessr Scoring:**
  - **Time — Gaussian Label Smoothing ($\sigma = 2.0$ yrs):** Gives smooth partial credit to adjacent years instead of hard one-hot cross-entropy.
  - **Space — GPU Geodesic Distance Soft Labels ($\tau_{\text{geo}} = 135\text{ km}$):** Computes exact spherical distance $d_{ij}$ from each photo's true `(lat, lon)` to all `1,249` tile centroids and sets $q(j \mid i) = \text{softmax}(-(d_{ij} - d_{i,\min}) / \tau_{\text{geo}})$. Naturally handles unequal tile sizes so physically close neighboring tiles receive strong partial credit while distant ones receive near-zero credit.
- **Coordinate & Year Decoding:**
  - **Year:** Probability-weighted expected year $\sum_k p_k (1900 + k)$.
  - **Location:** **Top-5 probability-weighted spherical centroid** (upgraded from Top-1 after gaining `+45` to `+84` free points on held-out Timeguessr).

---

## 4. Two-Stage Training Strategy (`04a_streetclip.ipynb`, `04b_siglip2.ipynb`)
1. **Stage 1 — Cached-Embedding Linear Probe:** Extract frozen ViT embeddings once in `bfloat16` to `data/embeddings/` (reusable by Notebook 02 FAISS KNN), then train the dual MLP heads in seconds per epoch.
2. **Stage 2 — Warm-Started Partial Fine-Tuning:** Initialize heads from Stage 1's best checkpoint (preventing random head gradients from corrupting pretrained ViT weights), unfreeze only the **last 6 transformer blocks**, and train end-to-end in `bfloat16` with gradient checkpointing, differential LRs (`2e-5` backbone / `3e-4` heads), crash-safe per-epoch auto-resume (`last_checkpoint.pt`), and **geolocation-safe augmentations** (mild crop + color jitter, **no horizontal flip** so street signs and driving sides stay intact).

---

## 5. Early Results, Error Analysis & Next Steps
- **Linear Probe Results (Held-Out Timeguessr, `n=1,600`):**
  - **StreetCLIP Linear Probe:** `7,241.9 / 10,000` mean (`7,757.9` median), `3,376.9` loc (`3,460.5` top-k), `3,865.0` year (`7.34` yr MAE, `88.4%` within ±15 yrs), `43.9%` country accuracy.
  - **SigLIP 2 Linear Probe:** `6,382.3 / 10,000` mean (`6,611.2` median), `2,784.9` loc (`2,829.8` top-k), `3,597.4` year (`8.14` yr MAE, `85.5%` within ±15 yrs), `32.5%` country accuracy.
- **Discretization vs. Misclassification Decomposition:** Only **5.6%–7.6%** (`123.7 pts`) of location score loss comes from tile discretization; **92%–94%** comes from tile misclassification (with `37.7%` of StreetCLIP guesses landing in the right country, wrong tile).
- **Key Bottleneck Discovered (2010s–2020s Year Bias):** StreetCLIP predicts `-13.7` yrs too early on 2010s photos and `-25.6` yrs too early on 2020s photos because almost all 2015–2026 training images are `OSV-5M` dashcam photos, teaching the time head a spurious *"not Street View → pre-2015"* shortcut.
- **Next Steps:**
  1. Add modern (2015–2026) non-Street-View photography to eliminate the temporal domain shift.
  2. Complete Stage 2 fine-tuning runs for both StreetCLIP and SigLIP 2 with the updated hyperparameters (`tau_geo_km=135`, `country_alpha=0.7`, top-5 spherical centroid decoding).
  3. Apply **EasyOCR + Lingua** country masking to the saved tile logits (`test_space_logits.npy`) in Notebook 05 and benchmark against a multimodal LLM.
