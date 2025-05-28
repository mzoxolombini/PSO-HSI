import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
import kagglehub
import os
from sklearn.decomposition import PCA
import cv2
import pandas as pd
import ace_tools_open as tools
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Load dataset
path = kagglehub.dataset_download("abhijeetgo/indian-pines-hyperspectral-dataset")
image = np.load(os.path.join(path, 'indianpinearray.npy'))
ground_truth = np.load(os.path.join(path, 'IPgt.npy'))

# Apply PCA to reduce to 1 band
pca = PCA(n_components=1)
reshaped = image.reshape(-1, image.shape[2])
image_pca = pca.fit_transform(reshaped).reshape(image.shape[0], image.shape[1])
image_pca = cv2.normalize(image_pca, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Fuzzy Entropy Objective Function
def fuzzy_entropy(thresholds, image):
    thresholds = np.sort(np.round(thresholds).astype(np.uint8))
    levels = np.concatenate(([0], thresholds, [255]))
    fuzzy_ent = 0
    for i in range(len(levels)-1):
        region = image[(image >= levels[i]) & (image < levels[i+1])]
        if region.size == 0:
            continue
        P = region / 255.0
        entropy = -np.sum(P * np.log(P + 1e-10) + (1 - P) * np.log(1 - P + 1e-10))
        fuzzy_ent += entropy
    return -fuzzy_ent

# Improved Particle Swarm Optimization (IPSO)
def ipso(objective, image, num_thresholds=5, num_particles=30, w=0.7, c1=1.5, c2=1.5):
    dim = num_thresholds
    lb, ub = 0, 255
    max_iter = dim * 1000
    X = np.random.uniform(lb, ub, (num_particles, dim))
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_scores = np.array([objective(p, image) for p in pbest])
    gbest = pbest[np.argmin(pbest_scores)]
    gbest_score = np.min(pbest_scores)
    for t in range(min(max_iter, 100)):
        mean_pos = np.mean(X, axis=0)
        for i in range(num_particles):
            r1, r2, r3 = np.random.rand(dim), np.random.rand(dim), np.random.rand(dim)
            V[i] = (w * V[i]
                    + c1 * r1 * (pbest[i] - X[i])
                    + c2 * r2 * (gbest - X[i])
                    + 0.5 * r3 * (mean_pos - X[i]))
            X[i] = np.clip(X[i] + V[i], lb, ub)
            score = objective(X[i], image)
            if score < pbest_scores[i]:
                pbest[i] = X[i]
                pbest_scores[i] = score
        if np.min(pbest_scores) < gbest_score:
            gbest = pbest[np.argmin(pbest_scores)]
            gbest_score = np.min(pbest_scores)
    return np.sort(np.round(gbest).astype(np.uint8)), -gbest_score

# Run IPSO and collect stats for L in [10, 12, 14]
def run_ipso_statistics(image, ground_truth, levels=[10, 12, 14], runs=3):
    results = []
    for L in levels:
        entropies, times, psnrs, ssims = [], [], [], []
        for _ in range(runs):
            start = time.time()
            thresholds, entropy = ipso(fuzzy_entropy, image, num_thresholds=L)
            duration = time.time() - start
            levels_arr = np.concatenate(([0], thresholds, [255]))
            segmented = np.zeros_like(image)
            for i in range(len(levels_arr)-1):
                segmented[(image >= levels_arr[i]) & (image < levels_arr[i+1])] = i + 1
            entropies.append(entropy)
            times.append(duration)
            psnrs.append(psnr(image, segmented, data_range=255))
            ssims.append(ssim(image, segmented, data_range=255))
        results.append([
            L, "IPSO", round(np.mean(times), 3), round(np.mean(entropies), 3),
            round(np.std(entropies), 3), round(np.mean(psnrs), 3), round(np.mean(ssims), 3)
        ])
    df = pd.DataFrame(results, columns=["L", "Algo", "T (in sec)", "fmean", "fstd", "PSNR", "SSIM"])
    tools.display_dataframe_to_user(name="IPSO Auto-calculated Summary", dataframe=df)

# Execute
run_ipso_statistics(image_pca, ground_truth)
