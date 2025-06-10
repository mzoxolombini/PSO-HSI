import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score
from pathlib import Path
import warnings
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd

warnings.filterwarnings("ignore")

# --- PSO Parameters ---
n_particles = 30
n_iterations = 50
w = 0.7
c1 = 1.5
c2 = 1.5

# --- Fuzzy Entropy Fitness ---
def fuzzy_entropy(image, thresholds):
    thresholds = np.sort(thresholds)
    thresholds = np.concatenate(([0], thresholds, [256]))
    entropy = 0.0
    for i in range(len(thresholds) - 1):
        mask = (image >= thresholds[i]) & (image < thresholds[i + 1])
        region = image[mask]
        if len(region) == 0:
            continue
        p = region / 255.0
        e = -np.sum(p * np.log(p + 1e-12))
        entropy += e
    return entropy

# --- PSO Optimizer ---
def pso_segmentation(image, n_thresholds):
    image = image.astype(np.uint8).flatten()
    dim = n_thresholds
    lb, ub = 0, 255

    particles = np.random.randint(lb, ub, (n_particles, dim))
    velocities = np.random.randn(n_particles, dim)

    pbest = particles.copy()
    pbest_scores = np.array([fuzzy_entropy(image, p) for p in particles])
    gbest = pbest[np.argmax(pbest_scores)]
    gbest_score = np.max(pbest_scores)

    for _ in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            particles[i] = np.clip(particles[i] + velocities[i], lb, ub).astype(int)
            score = fuzzy_entropy(image, particles[i])
            if score > pbest_scores[i]:
                pbest[i] = particles[i]
                pbest_scores[i] = score
                if score > gbest_score:
                    gbest = particles[i]
                    gbest_score = score
    return np.sort(gbest)

# --- Threshold Application ---
def apply_thresholds(image, thresholds):
    thresholds = np.sort(np.concatenate(([0], thresholds, [256])))
    segmented = np.zeros_like(image)
    for i in range(len(thresholds) - 1):
        mask = (image >= thresholds[i]) & (image < thresholds[i + 1])
        segmented[mask] = i + 1
    return segmented

# --- Main Function ---
def main():
    data_dir = Path(r"C:\Users\mzoxo\OneDrive\Documents\data")
    mat_files = list(data_dir.glob("*.mat"))

    gt_files = {f.stem.lower(): f for f in mat_files if "_gt" in f.stem.lower()}
    img_files = [f for f in mat_files if "_gt" not in f.stem.lower()]

    baseline_metrics = {
        "indian_pines": {"OA": 0.1105, "Kappa": 0.0000},
        "salinas": {"OA": 0.1240, "Kappa": 0.0465},
        "paviau": {"OA": 0.1045, "Kappa": 0.0355}
    }

    results = []

    for img_path in img_files:
        dataset_name = img_path.stem.lower().replace("_corrected", "")
        dataset_title = dataset_name.replace("_", " ").title()
        print(f"\n=== Processing dataset: {dataset_title} ===")

        for m_thresholds in range(1, 15):
            print(f"-- Running for k = {m_thresholds} thresholds")

            img_data = loadmat(img_path)
            img_key = [k for k in img_data.keys() if not k.startswith('__')][0]
            img = img_data[img_key]

            reshaped = img.reshape(-1, img.shape[-1])
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(reshaped).reshape(img.shape[:2])
            pc1_norm = ((pc1 - pc1.min()) / (pc1.max() - pc1.min()) * 255).astype(np.uint8)

            thresholds = pso_segmentation(pc1_norm, m_thresholds)
            segmented = apply_thresholds(pc1_norm, thresholds)

            img_name_lower = img_path.stem.lower().replace('_corrected', '')
            gt_path = None
            for gt_stem, gt_file in gt_files.items():
                if img_name_lower in gt_stem:
                    gt_path = gt_file
                    break

            if gt_path:
                gt_data = loadmat(gt_path)
                gt_key = [k for k in gt_data.keys() if not k.startswith('__')][0]
                gt = gt_data[gt_key]
                if gt.shape == segmented.shape:
                    acc = accuracy_score(gt.flatten(), segmented.flatten())
                    kappa = cohen_kappa_score(gt.flatten(), segmented.flatten())
                    print(f"OA: {acc:.4f} | Kappa: {kappa:.4f}")
                else:
                    print(f"GT shape mismatch: {gt.shape} vs {segmented.shape}")
                    acc = kappa = 0
            else:
                print("Ground truth not found for:", img_path.stem)
                acc = kappa = 0

            psnr_score = psnr(pc1_norm, segmented.astype(np.uint8), data_range=segmented.max())
            ssim_score = ssim(pc1_norm, segmented.astype(np.uint8), data_range=segmented.max())

            results.append({
                "Dataset": dataset_title,
                "k": m_thresholds,
                "PSNR": round(psnr_score, 4),
                "SSIM": round(ssim_score, 4),
                "OA": round(acc, 4),
                "Kappa": round(kappa, 4)
            })

            # --- Display images for each k ---
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(pc1_norm, cmap='gray')
            plt.title("PCA Band")

            plt.subplot(1, 3, 2)
            plt.imshow(segmented, cmap='nipy_spectral')
            plt.title(f"Segmented (k={m_thresholds})")

            if gt_path and gt.shape == segmented.shape:
                plt.subplot(1, 3, 3)
                plt.imshow(gt, cmap='nipy_spectral')
                plt.title("Ground Truth")

            plt.tight_layout()
            plt.show()

    results_df = pd.DataFrame(results)
    print("\nFinal Results:\n")
    print(results_df.to_string(index=True))

if __name__ == "__main__":
    main()
