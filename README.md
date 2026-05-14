# PSO-based Hyperspectral Image (HSI) Classification

## Description

This repository applies Particle Swarm Optimisation (PSO) and its variants — including PSO enhanced with Reinforcement Learning (RL) local search and an Improved multi-swarm IPSO — to multilevel thresholding and classification of hyperspectral remote sensing images.  
Fuzzy entropy is used as the thresholding fitness function, and classification is performed with Random Forest, Ensemble SVM (with SMOTE), and Composite Kernel SVM. Non-neural SOTA baselines (EMAP+SVM, LBP+SVM, SVM-CK) are also included for comparison.

---

## Repository Structure

| File | Purpose |
|------|---------|
| `pso_multilevel_thresholding.py` | Standard PSO-based multilevel thresholding for HSI classification using fuzzy entropy, PCA, and texture features (GLCM) evaluated with a Random Forest classifier. |
| `pso_fuzzy_entropy_segmentation.py` | PSO-only segmentation using fuzzy entropy optimisation on PCA-reduced hyperspectral data; focused on feature extraction and entropy scoring (no RL). |
| `pso_rl_local_search_classification.py` | PSO enhanced with a Reinforcement Learning agent that adaptively selects local search strategies (hill climbing variants); evaluated with an Ensemble SVM classifier and SMOTE oversampling. |
| `baseline_comparisons_emap_lbp_svm.py` | Non-neural SOTA baseline comparisons: EMAP+SVM, LBP+SVM, and SVM with Composite Kernel (SVM-CK). |
| `ipso_emap_composite_kernel_svm.py` | Improved IPSO (multi-swarm, context parameter, stagnation handling) with EMAP feature extraction and Composite Kernel SVM classification on the Indian Pines dataset. |

---

## Datasets

Three publicly available hyperspectral datasets are used:

| Dataset | Spatial Size | Spectral Bands | Classes |
|---------|-------------|----------------|---------|
| Indian Pines | 145 × 145 | 200 (corrected: 200 after removing water-absorption bands) | 16 |
| Salinas | 512 × 217 | 204 | 16 |
| PaviaU | 610 × 340 | 103 | 9 |

All datasets are provided by the [University of the Basque Country (EHU)](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).  
Most scripts auto-download the required `.mat` files into a `data/` directory on first run.

> **Note:** `baseline_comparisons_emap_lbp_svm.py` currently expects the `.mat` files to be present in a local path; update the `data_dir` variable in `main()` to point to your local `data/` folder.

---

## Dependencies

Install all required packages with:

```bash
pip install numpy scipy scikit-learn scikit-image matplotlib pandas imbalanced-learn numba tqdm requests
```

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `scipy` | `.mat` file loading, ndimage filters |
| `scikit-learn` | PCA, SVM, Random Forest, metrics |
| `scikit-image` | Image processing, GLCM, LBP, morphology |
| `matplotlib` | Visualisation |
| `pandas` | Results export to CSV |
| `imbalanced-learn` | SMOTE oversampling |
| `numba` | JIT-compiled fuzzy entropy (IPSO script) |
| `tqdm` | Progress bars |
| `requests` | HTTP downloads |

---

## Usage

Run each script directly from the repository root:

```bash
# Standard PSO multilevel thresholding (IndianPines, Salinas, PaviaU)
python pso_multilevel_thresholding.py

# PSO fuzzy entropy segmentation (IndianPines, Salinas, PaviaU)
python pso_fuzzy_entropy_segmentation.py

# PSO + RL local search classification (IndianPines, Salinas, PaviaU)
python pso_rl_local_search_classification.py

# Non-neural baseline comparisons (update data_dir in main() first)
python baseline_comparisons_emap_lbp_svm.py

# Improved IPSO with EMAP + Composite Kernel SVM (Indian Pines only)
python ipso_emap_composite_kernel_svm.py
```

Datasets are downloaded automatically to `data/` on first run (except `baseline_comparisons_emap_lbp_svm.py`).

---

## Methods

### PSO (Particle Swarm Optimisation)
A population-based meta-heuristic where particles explore the solution space guided by their personal best position and the global best. Used here to search for optimal multilevel thresholds that maximise fuzzy entropy.

### Fuzzy Entropy
A thresholding criterion that measures the uncertainty/information content of pixel intensity distributions within each segmented region. Higher fuzzy entropy indicates better-separated segments.

### EMAP (Extended Multi-Attribute Profiles)
Spatial feature extraction through a sequence of morphological operations (opening, closing, dilation, erosion) applied at multiple scales to the principal components of the hyperspectral image.

### LBP (Local Binary Pattern)
A texture descriptor that encodes the local structure around each pixel by comparing it with its neighbours, producing a rotation-invariant feature vector.

### SVM-CK (SVM with Composite Kernel)
An SVM classifier whose kernel combines a spatial (RBF on spatial features) and a spectral (RBF on spectral features) kernel, capturing complementary information for HSI classification.

### RL-Guided Local Search
A Q-learning agent that maintains Q-values for five hill climbing variants (simple, steepest ascent, stochastic, first-choice, random restart) and selects the most rewarding strategy at each PSO iteration.

### IPSO (Improved PSO)
A multi-swarm extension of PSO where each threshold dimension is optimised by an independent swarm. A context parameter aggregates the best positions across swarms; stagnation is handled by replacing the worst particles with mutated copies of the best ones.

---

## Metrics

| Metric | Description |
|--------|-------------|
| **OA** (Overall Accuracy) | Fraction of correctly classified pixels |
| **Kappa** | Cohen's Kappa coefficient (agreement beyond chance) |
| **Mean Accuracy (MA)** | Average per-class accuracy |
| **PSNR** | Peak Signal-to-Noise Ratio between original and segmented image |
| **SSIM** | Structural Similarity Index between original and segmented image |
| **IoU** | Mean Intersection over Union across all classes |
