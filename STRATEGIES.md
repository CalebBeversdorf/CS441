# Timeguessr AI — Strategies & Design Decisions

A concise summary of the data, discretization, modeling, training, evaluation, and interactive deployment strategies for **Timeguessr AI** (predicting a photo's capture year `1900–2026` and geographic coordinates `(lat, lon)` scored out of `10,000` points).

---

## 1. Data Curation & Leak-Free Splitting (`01_data_*.ipynb`)
- **Rebalanced Training/Validation Pool (`96,640` external images across `239` countries):** Iterated from an initial `170,945`-image pool (30.3% Russia, almost no modern non-Street-View photos) to a cleaner, better-balanced 6-source dataset:
  - `OSV-5M` (`25,000`, downsampled from `46,461` to prevent Street-View shortcuts)
  - `YFCC15M` (`22,281`, global Flickr photos, ~2000–2014) + `YFCC-Pre2000` (`12,851`, pre-2000)
  - `PastVu` (`12,016`, downsampled from `50,000`) + `PastVu-NonRU` (`15,000`, targeted non-Russian historical photos — cutting Russia's share from `30.3% → 13.2%`)
  - **`Wikimedia Commons` (`9,492`, 2015–2026, median 2020):** Added modern non-Street-View photography specifically to eliminate the 2010s–2020s temporal domain shift.
- **Zero-Leakage Evaluation:** Split external sources into `87,056` train / `9,584` val, and kept **`1,600` Timeguessr daily challenge photos** strictly held out for final testing (never used for tiling, training, early stopping, or hyperparameter selection).
- **Sublinear Country × Decade Balanced Sampling:** Uses a `WeightedRandomSampler` with sublinear weights $w_i \propto n_{\text{country}}^{\alpha_c - 1} \cdot n_{\text{decade}}^{\alpha_d - 1}$ ($\alpha_c = 0.7$, $\alpha_d = 0.75$) to curb geographic and temporal skew without over-boosting tiny countries.

---

## 2. World Discretization (`02_s2_world_tiling.ipynb`)
Direct `(lat, lon)` regression collapses toward ocean/continental averages under visual ambiguity, so we frame geolocation as classification over discrete tiles. Comparison on the held-out Timeguessr oracle ceiling (rebalanced dataset):

| Partition | Tiles | Country Purity | Oracle Med Dist | Oracle Mean Loc Score (/5k) |
| :--- | :---: | :---: | :---: | :---: |
| `s2_coarse` (Adaptive S2 Quadtree) | 379 | 43.0% | 53.1 km | 4,623.8 |
| `s2_medium` (Data Spherical Centroid) | 888 | 65.5% | 11.6 km | 4,800.4 |
| `s2_medium` (Geometric S2 Center — ablation) | 888 | 65.5% | 65.5 km | 4,530.7 |
| `s2_fine` (Adaptive S2 Quadtree) | 1,526 | 75.8% | 5.3 km | 4,846.2 |
| **`semantic_medium` (Country-Aware Spherical K-Means)** | **841** | **100.0%** | **9.5 km** | **4,851.6** |

**Key findings:**
- Replacing S2 geometric cell centers with **data-weighted 3D spherical centroids** cut median oracle error by **5.6×** (`65.5 km → 11.6 km`) and gained `+269.7 pts`.
- **`semantic_medium`** (clustering *within* each country via Spherical K-Means) achieved the **highest oracle ceiling (`4,851.6`)** with fewer tiles than `s2_medium`, aligned tiles with cities/borders, and guaranteed **100% single-country purity** for downstream OCR language masking.

---

## 3. Dual-Head ViT Modeling, Losses & Infrastructure (`src/timeguessr_ai/`)
- **Backbone Comparison:**
  - **`geolocal/StreetCLIP`** (`ViT-L/14@336px`, 24 blocks, ~304M vision params) — CLIP pretrained specifically on geotagged street-level imagery.
  - **`google/siglip2-base-patch16-384`** (`ViT-B/16@384px`, 12 blocks, ~86M vision params) — modern general-purpose vision-language encoder with higher input resolution and MAP attention pooling.
- **Dual MLP Heads:** Shared vision embedding (`LayerNorm`) feeds a **Time Head** (`127` classes for `1900–2026`) and a **Space Head** (`841` `semantic_medium` tiles).
- **Soft-Label Losses Aligned with Timeguessr Scoring:**
  - **Time — Gaussian Label Smoothing ($\sigma = 2.0$ yrs):** Smooth partial credit to adjacent years instead of hard one-hot cross-entropy.
  - **Space — GPU Geodesic Distance Soft Labels ($\tau_{\text{geo}} = 135\text{ km}$):** Computes exact spherical distance $d_{ij}$ from each photo's true `(lat, lon)` to all `841` tile centroids and sets $q(j \mid i) = \text{softmax}(-(d_{ij} - d_{i,\min}) / \tau_{\text{geo}})$. Naturally handles unequal tile sizes so physically close neighboring tiles receive strong partial credit.
- **Coordinate & Year Decoding:**
  - **Year:** Probability-weighted expected year $\sum_k p_k (1900 + k)$.
  - **Location:** **Top-5 probability-weighted spherical centroid** (primary decoder; gains `+25` to `+55` free points over Top-1 across runs).
- **Cross-Platform Accelerator Support:** `pyproject.toml` (`uv sync --extra cu128|rocm|xpu|cpu`) and `src/timeguessr_ai/train.py` auto-select `cuda` (NVIDIA / AMD ROCm) → `xpu` (Intel) → `mps` (Apple Silicon) → `cpu`.

---

## 4. Two-Stage Training & Sweeping (`03a_streetclip.ipynb`, `03b_siglip2.ipynb`)
1. **Stage 1 — Cached-Embedding Linear Probe:** Extract frozen ViT embeddings once in `bfloat16` to `data/embeddings/` with **SHA-256 `image_path` manifest verification** (`load_cached_embeddings`), then train the dual MLP heads in ~2s/epoch (~50s per 25-epoch run).
2. **Stage 2 — Warm-Started Partial Fine-Tuning:** Warm-start heads from Stage 1's best checkpoint, unfreeze the **last 6 transformer blocks**, and fine-tune end-to-end in `bfloat16` with gradient checkpointing, differential LRs (`2e-5` backbone / `3e-4` heads), crash-safe per-epoch auto-resume (`last_checkpoint.pt`), and **geolocation-safe augmentations** (mild crop + color jitter, **no horizontal flip**).
3. **Hyperparameter & Post-Hoc Decoding Sweeps (`03a_streetclip.ipynb`):**
   - **Stage 1 Coordinate Sweep:** Fast 1D sweep on cached embeddings over `tau_geo_km`, `sigma_year`, `country_alpha`, `dropout`, `weight_decay`, and `lp_lr` (selecting strictly on `val_total_score`), followed by a combined-best check.
   - **Zero-Retraining Post-Hoc Decoding Sweep:** Sweeps `topk_space` × `T_space` and `year_mode` (`expected` vs. `argmax`) × `T_time` on validation logits, with heatmaps, delta bars, CDFs, and decade error analysis.

---

## 5. Interactive Playable Arena (`07_interactive_game.ipynb`, `game.py`, `predictors.py`)
- Built a unified **`BaseTimeguessrPredictor`** interface (`LinearProbePredictor`, `FineTunedPredictor`, with offline reverse geocoding via `reverse_geocoder`) so any model (ViT, FAISS KNN, OCR, or LLM) can plug directly into Notebook 07.
- Added **`src/timeguessr_ai/game.py`** to fetch live TimeGuessr rounds and all-player score histograms/percentiles for head-to-head human vs. AI play.

---

## 6. Current Results, Error Analysis & Next Steps (`03c_model_comparison.ipynb`)

### Held-Out Timeguessr Benchmark (`n = 1,600`)
| Model | Total Mean (/10k) | Total Median | Loc Mean (/5k) | Year Mean (/5k) | Med Dist (km) | Country Acc | Year MAE | ≤15 yr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **StreetCLIP — Fine-Tune (Last 6)** | **7,684.6** | **8,167.5** | **3,457.4** | **4,227.2** | **365.0** | 43.3% | **4.82** | **96.4%** |
| StreetCLIP — Linear Probe | 7,616.6 | 8,099.9 | 3,445.0 | 4,171.6 | 376.9 | **44.1%** | 5.20 | 95.1% |
| SigLIP 2 — Linear Probe | 6,565.5 | 6,864.6 | 2,867.1 | 3,698.4 | 679.5 | 35.6% | 7.53 | 88.7% |

- **Data Rebalancing & Wikimedia Payoff:** Rebalancing the dataset and adding `9,492` modern Wikimedia Commons photos boosted StreetCLIP Linear Probe by **`+374.7 pts`** (`7,241.9 → 7,616.6`) and SigLIP 2 Linear Probe by **`+183.2 pts`** (`6,382.3 → 6,565.5`).
- **2010s–2020s Temporal Bias Fixed:** Previously, StreetCLIP predicted `-13.7 yrs` too early on 2010s and `-25.6 yrs` too early on 2020s due to OSV-5M Street-View shortcutting. On the fine-tuned model, 2010s bias dropped to **`+0.92 yrs`** (`2.70` yr MAE, `8,169` mean total) and 2020s bias dropped nearly 5× to **`-5.37 yrs`**, cutting overall year MAE from `7.34 → 4.82 yrs`.
- **Stage 2 Fine-Tuning Gains:** Unfreezing the last 6 StreetCLIP blocks (5 epochs, ~5.3 hrs on RTX 4070 SUPER) added **`+68.1 pts`** on held-out Timeguessr (`+55.6` year, `+12.4` loc), cutting year MAE from `5.20 → 4.82` and median distance from `376.9 → 365.0 km`.
- **Discretization vs. Misclassification Decomposition (`semantic_medium`, 841 tiles):** Only **9.5%–9.6%** (`148.4 pts`) of location score loss comes from tile discretization; **90.4%–90.5%** (`1,394–1,407 pts`) comes from tile misclassification (`36.8%–37.4%` land in the right country, wrong tile; `28.8%–29.1%` hit the exact tile in Top-5).

### Next Steps
Finish the remaining notebooks:
   - `04_baseline_faiss_knn.ipynb` — FAISS KNN retrieval baseline on cached embeddings.
   - `05_ocr_addon.ipynb` — **EasyOCR + Lingua** country masking applied to saved `test_space_logits.npy`.
   - `06_LLM_comparison.ipynb` — Multimodal LLM benchmark on the same held-out Timeguessr split.
