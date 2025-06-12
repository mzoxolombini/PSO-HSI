import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from pathlib import Path
import warnings
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import random
from skimage.feature import local_binary_pattern

warnings.filterwarnings("ignore")

# --- PSO Parameters ---
n_particles = 30
n_iterations = 50
w = 0.7
c1 = 1.5
c2 = 1.5

# --- Adaptive Local Search Strategy using RL (Q-Learning) ---
actions = ['greedy', 'random', 'annealing']
Q_table = {a: 0 for a in actions}
epsilon = 0.3
alpha = 0.1
gamma = 0.9

def choose_action():
    if random.random() < epsilon:
        return random.choice(actions)
    return max(Q_table, key=Q_table.get)

def update_Q(action, reward):
    best_future = max(Q_table.values())
    Q_table[action] += alpha * (reward + gamma * best_future - Q_table[action])

# --- Fisher Score Fitness ---
def fisher_score(image, thresholds):
    thresholds = np.sort(thresholds)
    thresholds = np.concatenate(([0], thresholds, [256]))
    scores = []
    global_mean = np.mean(image)
    for i in range(len(thresholds) - 1):
        mask = (image >= thresholds[i]) & (image < thresholds[i + 1])
        region = image[mask]
        if len(region) == 0:
            continue
        region_mean = np.mean(region)
        region_var = np.var(region) + 1e-6
        score = len(region) * ((region_mean - global_mean) ** 2) / region_var
        scores.append(score)
    return np.sum(scores)

# --- Local Search Variants ---
def greedy_hill_climbing(image, thresholds):
    best_thresholds = thresholds.copy()
    best_score = fisher_score(image, best_thresholds)
    for i in range(len(thresholds)):
        for delta in [-1, 1]:
            new_thresholds = best_thresholds.copy()
            new_thresholds[i] = np.clip(new_thresholds[i] + delta, 1, 254)
            new_score = fisher_score(image, new_thresholds)
            if new_score > best_score:
                best_thresholds = new_thresholds
                best_score = new_score
    return np.sort(best_thresholds), best_score

def random_local_search(image, thresholds):
    best_thresholds = thresholds.copy()
    best_score = fisher_score(image, best_thresholds)
    for _ in range(10):
        new_thresholds = best_thresholds + np.random.randint(-5, 6, size=len(thresholds))
        new_thresholds = np.clip(new_thresholds, 1, 254)
        new_score = fisher_score(image, new_thresholds)
        if new_score > best_score:
            best_thresholds = new_thresholds
            best_score = new_score
    return np.sort(best_thresholds), best_score

def simulated_annealing(image, thresholds):
    current = thresholds.copy()
    current_score = fisher_score(image, current)
    T = 10.0
    T_min = 1e-3
    alpha = 0.95
    while T > T_min:
        i = np.random.randint(len(current))
        candidate = current.copy()
        candidate[i] = np.clip(candidate[i] + np.random.choice([-1, 1]), 1, 254)
        score = fisher_score(image, candidate)
        if score > current_score or np.exp((score - current_score) / T) > np.random.rand():
            current = candidate
            current_score = score
        T *= alpha
    return np.sort(current), current_score

# --- Adaptive Local Search Controller ---
def rl_local_search(image, thresholds):
    action = choose_action()
    if action == 'greedy':
        refined, score = greedy_hill_climbing(image, thresholds)
    elif action == 'random':
        refined, score = random_local_search(image, thresholds)
    else:
        refined, score = simulated_annealing(image, thresholds)

    reward = score - fisher_score(image, thresholds)
    update_Q(action, reward)
    return refined

# --- PSO Optimizer ---
def pso_segmentation(image, n_thresholds):
    image = image.astype(np.uint8).flatten()
    dim = n_thresholds
    lb, ub = 0, 255

    particles = np.random.randint(lb, ub, (n_particles, dim))
    velocities = np.random.randn(n_particles, dim)

    pbest = particles.copy()
    pbest_scores = np.array([fisher_score(image, p) for p in particles])
    gbest = pbest[np.argmax(pbest_scores)]
    gbest_score = np.max(pbest_scores)

    for _ in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            particles[i] = np.clip(particles[i] + velocities[i], lb, ub).astype(int)
            score = fisher_score(image, particles[i])
            if score > pbest_scores[i]:
                pbest[i] = particles[i]
                pbest_scores[i] = score
                if score > gbest_score:
                    gbest = particles[i]
                    gbest_score = score

    refined = rl_local_search(image, gbest)
    return np.sort(refined)

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
            pca = PCA(n_components=3)
            pc_img = pca.fit_transform(reshaped).reshape(*img.shape[:2], 3)
            lbp = np.mean([local_binary_pattern(pc_img[:, :, i], P=8, R=1, method="uniform") for i in range(3)], axis=0)
            fused = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)

            thresholds = pso_segmentation(fused, m_thresholds)
            segmented = apply_thresholds(fused, thresholds)

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

            psnr_score = psnr(fused, segmented.astype(np.uint8), data_range=segmented.max())
            ssim_score = ssim(fused, segmented.astype(np.uint8), data_range=segmented.max())

            results.append({
                "Dataset": dataset_title,
                "k": m_thresholds,
                "PSNR": round(psnr_score, 4),
                "SSIM": round(ssim_score, 4),
                "OA": round(acc, 4),
                "Kappa": round(kappa, 4)
            })

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(fused, cmap='gray')
            plt.title("LBP-Fused")

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

    for metric in ['OA', 'Kappa', 'PSNR', 'SSIM']:
        plt.figure()
        for dataset in results_df['Dataset'].unique():
            subset = results_df[results_df['Dataset'] == dataset]
            plt.plot(subset['k'], subset[metric], label=dataset)
        plt.xlabel('k (Thresholds)')
        plt.ylabel(metric)
        plt.title(f'{metric} vs Thresholds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
