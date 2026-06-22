# Generative Data Augmentation for WiFi Fingerprint Localization Under Spatial Sparsity

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006400.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-blueviolet.svg)](https://optuna.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Colab Ready](https://img.shields.io/badge/Colab-Ready-F9AB00.svg)](https://colab.research.google.com/)

`wifi-fingerprinting` `indoor-localization` `generative-models` `data-augmentation` `spatial-sparsity` `VAE` `WGAN-GP` `diffusion-models` `XGBoost` `deep-learning` `radio-map` `RSSI` `signal-processing` `machine-learning` `indoor-positioning`

---

This repository investigates how **spatial sparsity** degrades WiFi fingerprint-based indoor localization and whether **generative data augmentation** can rescue positioning accuracy when reference data is scarce.

## Problem Statement

WiFi fingerprint localization relies on dense radio maps: RSSI (Received Signal Strength Indicator) vectors collected at many known reference points across an indoor environment. At each location, a device scans all visible WiFi access points and records the signal strength from each, forming a high-dimensional fingerprint that encodes the spatial characteristics of that position. A regression model trained on these fingerprints can then predict the coordinates of a new device from its observed RSSI vector alone.

In practice, constructing these radio maps is labor-intensive and expensive: a surveyor must physically visit each reference point, often multiple times, with calibrated equipment. When the survey budget is limited, reference locations become spatially sparse, and the resulting radio map develops coverage gaps. The regression model, trained on this incomplete data, cannot interpolate accurately into the unmapped regions, leading to significant localization error.

This project systematically quantifies that degradation across multiple models and datasets, and proposes a generative augmentation pipeline to synthesize fill-in fingerprints that bridge the spatial gaps.

## Methodology

The project is structured as a two-phase study:

### Phase 1: Sparsification Ablation Study

The first phase isolates the effect of spatial sparsity on localization accuracy. Starting from a fully surveyed radio map, we progressively remove reference locations and measure the impact on positioning error.

#### 1.1 Farthest-Point Thinning

Random removal of reference locations would produce uneven gaps: some regions could lose all nearby references while others remain dense. Instead, we employ **farthest-point thinning**, a deterministic spatial subsampling strategy that guarantees uniform coverage reduction.

The algorithm works as follows:
1. Compute the centroid of all unique reference locations.
2. Select the location closest to the centroid as the first point.
3. Iteratively select the location farthest from all previously selected points (maximizing the minimum distance to the existing set).
4. Continue until all locations are ordered.

The resulting ordering ranks locations by their contribution to spatial coverage: points selected first form the most spread-out skeleton, while points selected last are dense fill-ins closest to already-covered areas. To simulate X% omission, we drop the last X% of this ordering, preserving the most spatially informative locations and thinning uniformly across the floor plan.

Omission levels tested: **10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%**.

#### 1.2 Feature Engineering

Each model is evaluated under two feature preparation tracks to assess whether feature selection interacts with sparsity:

- **Full track**: All access points (APs) with >5% coverage across training samples are retained. Only completely dead APs (those heard by fewer than 5% of training samples) are removed. RSSI values are normalized to [0, 1] using MinMaxScaler fitted on the training set, with no-signal values first mapped to -105 dBm.

- **Selected track**: After dead-AP removal, an `ExtraTreesRegressor` (200 trees) is fitted on the training data, and `SelectFromModel` with a median importance threshold is applied. This retains only the APs most predictive of position, typically reducing dimensionality by ~50%.

#### 1.3 Regression Models

Four regression architectures spanning different model families are evaluated:

| Model | Architecture | Hyperparameter Search Space |
|-------|-------------|---------------------------|
| **kNN** | k-Nearest Neighbors | k: [1, 15], weights: {uniform, distance} |
| **SVR** | Support Vector Regression (RBF kernel) via MultiOutputRegressor | C: {0.1, 1, 10, 100}, gamma: {scale, 0.01, 0.1} |
| **XGBoost** | Gradient Boosted Trees via MultiOutputRegressor | n_estimators: [100, 300], max_depth: {3, 5, 7}, learning_rate: [0.01, 0.1] (log) |
| **DNN** | 3-layer neural network (256 &rarr; 128 &rarr; 64) with BatchNorm, ReLU, Dropout | learning_rate: [1e-4, 1e-3] (log), dropout: [0.1, 0.3] |

Hyperparameters are optimized with **Optuna** using the TPE (Tree-structured Parzen Estimator) sampler with 10 trials per model-track-sparsity configuration. The DNN uses cosine annealing learning rate scheduling and early stopping with patience of 30 epochs on validation MEE.

#### 1.4 Evaluation Protocol

- **Fixed 60/20/20 split**: Stratified by location. The train/val/test partition is created once on the full dataset and held constant.
- **Sparsity applied to train+val only**: When locations are omitted, only training and validation samples at those locations are removed. The test set always uses the full dense grid, ensuring all models are evaluated against the same ground truth regardless of training sparsity.
- **Metric**: Mean Euclidean Error (MEE) in meters between predicted and true coordinates, plus P50 (median) and P90 (90th percentile) error.

#### 1.5 Cross-Dataset Replication

The entire Phase 1 study is replicated independently across three indoor environments:

| Dataset | Source | Samples | APs | Unique Locations | No-Signal Value |
|---------|--------|--------:|----:|------------------:|:------:|
| **UTS** | UTSIndoorLoc | 522 | 206 | 198 | 100 |
| **UJI Bldg 1** | UJIndoorLoc | 1,577 | 149 | 73 | -105 |
| **UJI Bldg 2** | UJIndoorLoc | 1,396 | 190 | 79 | -105 |

These datasets differ substantially in scale, AP density, number of reference locations, and physical environment, providing a strong test of generalizability.

---

### Phase 2: Generative Rescue Pipeline

The second phase tests whether generative models trained on sparse data can synthesize realistic WiFi fingerprints to fill coverage gaps and improve the downstream regressor.

Based on Phase 1 results, **XGBoost** is selected as the downstream regressor (best performance at 0% omission: MEE = 1.66 m on UTS). The rescue pipeline is evaluated at sparsity levels where degradation is most severe: **30%, 50%, 70%, 80%, 90%** omission.

#### 2.1 Generative Architectures

Six generative models spanning three major paradigm families are implemented:

| Model | Family | Architecture | Reference |
|-------|--------|-------------|-----------|
| **VAE** | Variational Inference | Encoder (256 &rarr; 128) &rarr; latent (64-dim, reparameterization trick) &rarr; Decoder (128 &rarr; 256 &rarr; output). KL divergence weight = 0.001. | Suroso et al., ECTI-CON 2022 |
| **GAN** | Adversarial | WGAN-GP: Generator with BatchNorm (256 &rarr; 512 &rarr; 512 &rarr; output + Sigmoid), Critic with Spectral Normalization + Minibatch Discrimination (512 &rarr; 256 &rarr; 1, no sigmoid). 5 critic steps per generator step. Gradient penalty weight = 10. | Njima et al., IEEE Access 2022 (improved) |
| **DDPM** | Diffusion (MLP) | Flat MLP noise predictor (input + time embedding &rarr; 512 hidden &rarr; output). Sinusoidal positional encoding for timestep conditioning. Self-conditioning branch: concatenates x0 estimate with x_t. 1000 diffusion steps, linear beta schedule. EMA (decay = 0.999). | Guo et al., CCDC 2023 |
| **DIT** | Diffusion (Transformer) | Diffusion Transformer that patchifies the RSSI vector into tokens (patch_size APs each projected to hidden_dim). Multi-head self-attention with sinusoidal time embedding broadcast to all tokens. | Yan et al., SPIE 2025 |
| **Latent DIT** | Latent Diffusion (Transformer) | **Proposed**. Combines a VAE encoder/decoder with a DIT operating in the compressed latent space. The VAE compresses the high-dimensional RSSI vector into a lower-dimensional latent, where the DIT performs diffusion. Decoding maps samples back to RSSI space. | Proposed |
| **TDPM** | Tabular Diffusion | Tabular Diffusion Probabilistic Model designed for structured/tabular data. | Zhumagali et al., 2025 |

All models include a **downstream validation hook**: every N training epochs, the model generates a probe batch of synthetic samples, which are filtered, pseudo-labelled, combined with sparse real data, used to retrain XGBoost, and evaluated on the held-out validation set. The checkpoint with the lowest downstream validation MEE is restored for final generation.

#### 2.2 Quality Filtering

Raw generative output often contains unrealistic samples (e.g., too many active APs, RSSI patterns unlike any real observation). A two-stage quality gate filters these before use:

**Stage 1 &mdash; Sparsity Mask**: Enforces realistic AP activation patterns. For each synthetic sample, the number of active APs k is drawn from the empirical distribution of real training data (Gaussian fit on real active AP counts, clipped to [min_real, P95_real]). Only the top-k strongest signals are kept; remaining APs are zeroed. This ensures synthetic fingerprints have the same sparsity structure as real WiFi observations.

**Stage 2 &mdash; KNN Gate**: Rejects out-of-distribution samples. For each synthetic fingerprint, the mean distance to its k=5 nearest neighbors in the real training set is computed. A threshold is set at the 90th percentile of the same metric computed among real-to-real samples. Synthetic samples exceeding this threshold are discarded as too far from the learned RSSI manifold.

Typical pass rates after both stages: **50-80%** of generated samples survive.

#### 2.3 Pseudo-Labelling

Filtered synthetic fingerprints lack position labels (the generative models only learn the RSSI distribution, not the mapping to coordinates). An XGBoost regressor trained on the sparse real data assigns (x, y) **pseudo-labels** to each surviving synthetic sample. This is equivalent to asking: "where would this fingerprint most likely be observed, according to our best model of the sparse environment?"

#### 2.4 Augmentation and Retraining

The pseudo-labelled synthetic data is concatenated with the original sparse real training data, and XGBoost is retrained from scratch on the combined dataset. The augmented model is evaluated on the same fixed test set used throughout the study, enabling direct comparison against the sparse-only baseline.

---

## Key Results

### Sparsification: Error Degradation Across Models

All models degrade monotonically with increasing sparsity, but at very different rates. XGBoost and kNN consistently outperform SVR and DNN.

![Sparsification Degradation](assets/sparsification_degradation.png)

**MEE (m) for all models at each sparsity level (UTS, Full features):**

| Model | 0% | 10% | 20% | 30% | 50% | 70% | 90% |
|-------|---:|----:|----:|----:|----:|----:|----:|
| **kNN** | 2.50 | 2.69 | 2.92 | 4.42 | 4.91 | 5.93 | **9.03** |
| **SVR** | 6.62 | 6.77 | 6.97 | 7.13 | 7.11 | 8.60 | 11.59 |
| **XGBoost** | **1.66** | **2.33** | **2.78** | **3.53** | 4.30 | 7.13 | 9.96 |
| **DNN** | 3.78 | 3.83 | 5.11 | 4.48 | 5.00 | 8.84 | 14.75 |

### Cross-Dataset Heatmaps

The pattern is consistent across all three datasets: errors scale super-linearly beyond ~50% omission.

![Heatmap All Datasets](assets/heatmap_all_datasets.png)

### Relative Degradation Across Datasets

Even though absolute error levels differ, the *relative* degradation pattern is strikingly similar across environments. The best model's MEE increases 5-13x at 90% omission.

![Cross-Dataset Degradation](assets/cross_dataset_degradation.png)

### Generative Rescue: Can Synthetic Data Help?

At moderate sparsity (30-70%), generative augmentation provides meaningful improvements. GAN shows the strongest gains at 30% omission, while VAE and Latent DIT perform best at extreme sparsity (80-90%).

![Generative Rescue Curves](assets/generative_rescue_curves.png)

### Improvement Heatmap

Green cells indicate where augmentation beats the baseline; red cells indicate where it hurts.

![Improvement Heatmap](assets/generative_improvement_heatmap.png)

**Key findings from the generative rescue:**
- At **30% omission**, GAN reduces MEE by 0.50 m (12% improvement), and all models except DIT help
- At **70% omission**, GAN, VAE, and DDPM all reduce MEE, with GAN delivering -0.78 m
- At **80% omission**, GAN (-0.89 m) and VAE (-0.47 m) still improve over baseline
- At **90% omission**, Latent DIT (-0.85 m) and VAE (-0.68 m) provide the best rescue
- No single generative model dominates at all sparsity levels

### Model Comparison by Sparsity Level

![Bar Comparison](assets/generative_bar_comparison.png)

## Repository Structure

```
wifi_localization/
├── FINAL_sparsification_study_UTS.ipynb      # Phase 1: Sparsification study (UTS)
├── sparsification_uts-2.ipynb                # Phase 1: Sparsification study (UTS, variant)
├── sparsification_uj1-2.ipynb                # Phase 1: Sparsification study (UJI Building 1)
├── sparsification_uj2-2.ipynb                # Phase 1: Sparsification study (UJI Building 2)
├── augmentation_experiment_UTS_1.ipynb        # Phase 2: Generative rescue pipeline (UTS)
├── generative_rescue_9.ipynb                  # Phase 2: Generative rescue pipeline (UTS, full)
├── generative_rescue_uts_2.ipynb              # Phase 2: Generative rescue pipeline (UTS, v2)
├── generative_rescue_uj2_2.ipynb              # Phase 2: Generative rescue pipeline (UJI Bldg 2)
├── generative_rescue_results_reconstructed.json  # Aggregated rescue results
├── uts_dataset.csv                            # UTS indoor localization dataset
├── uj_building1.csv                           # UJIndoorLoc Building 1
├── uj_building2.csv                           # UJIndoorLoc Building 2
├── ResultsPoster.pdf                          # Research poster
├── assets/                                    # Plots and figures
└── LICENSE                                    # Apache 2.0
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- scikit-learn
- XGBoost
- Optuna
- matplotlib, numpy, pandas, scipy

All notebooks are designed to run on Google Colab with GPU acceleration.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
