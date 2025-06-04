import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import disk, opening, closing
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter
import os
import urllib.request
from scipy.io import loadmat
from skimage.util import img_as_ubyte
from skimage.filters.rank import gradient
import warnings
import datetime

warnings.filterwarnings("ignore")

def compute_fuzzy_entropy(gray, thresholds):
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    hist = np.cumsum(hist) / np.sum(hist)
    x = np.arange(256)
    total_entropy = 0.0

    thresholds = [0] + thresholds + [255]
    for k in range(1, len(thresholds)):
        a, b = thresholds[k - 1], thresholds[k]
        delta = max(b - a, 1e-10)
        mu = np.clip((x - a) / delta, 0, 1) * np.clip((b - x) / delta, 0, 1)
        R_k = np.sum(hist * mu)
        if R_k > 1e-10:
            terms = (hist * mu) / R_k
            terms = np.clip(terms, 1e-10, 1)
            H_k = -np.sum(terms * np.log(terms))
            total_entropy += H_k

    return total_entropy

def local_search(gray, initial_thresholds, max_iter=30):
    best = initial_thresholds.copy()
    if isinstance(best[0], list):  # If it's a nested list, flatten it
        best = best[0]
    best_score = compute_fuzzy_entropy(gray, best[1:-1])
    for _ in range(max_iter):
        improved = False
        for i in range(1, len(best) - 1):
            for delta in [-1, 1]:
                neighbor = best.copy()
                neighbor[i] = np.clip(neighbor[i] + delta, best[i - 1] + 1, best[i + 1] - 1)
                score = compute_fuzzy_entropy(gray, neighbor[1:-1])
                if score > best_score:
                    best = neighbor.copy()
                    best_score = score
                    improved = True
        if not improved:
            break
    return best


def ipso_optimize(gray, k_levels, pop_size=30, max_iter=50):
    dim = k_levels - 1
    pop = np.sort(np.random.randint(1, 254, (pop_size, dim)), axis=1)
    vels = np.zeros_like(pop)
    pbest = pop.copy()
    pbest_scores = np.array([compute_fuzzy_entropy(gray, list(p)) for p in pbest])
    gbest_idx = np.argmax(pbest_scores)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]

    best_fitness_history = [gbest_score]

    for iteration in range(max_iter):
        w = 0.9 - iteration * (0.5 / max_iter)
        c1, c2 = 2.0, 2.0
        for i in range(pop_size):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vels[i] = w * vels[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (gbest - pop[i])
            pop[i] = np.clip(np.round(pop[i] + vels[i]), 1, 254).astype(int)
            pop[i] = np.sort(pop[i])

            score = compute_fuzzy_entropy(gray, list(pop[i]))
            if score > pbest_scores[i]:
                pbest[i] = pop[i].copy()
                pbest_scores[i] = score
                if score > gbest_score:
                    gbest = pop[i].copy()
                    gbest_score = score
        best_fitness_history.append(gbest_score)

    thresholds = [0] + list(gbest) + [255]
    refined = local_search(gray, thresholds, max_iter=20)
    return refined, gbest_score, best_fitness_history

def compute_emap(gray):
    gray_ubyte = img_as_ubyte((gray - gray.min()) / (gray.max() - gray.min()))
    profiles = []
    for selem_size in [1, 3, 5]:
        selem = disk(selem_size)
        profiles.append(opening(gray_ubyte, selem))
        profiles.append(closing(gray_ubyte, selem))
        profiles.append(gradient(gray_ubyte, selem))
    return np.stack(profiles, axis=-1)

def segment_image(gray, thresholds):
    segmented = np.zeros_like(gray, dtype=np.float32)
    gray_norm = gray.astype(np.float32) / 255.0
    for i in range(len(thresholds) - 1):
        lower = thresholds[i] / 255.0
        upper = thresholds[i + 1] / 255.0
        delta = max(upper - lower, 1e-10)
        mask = np.where(gray_norm <= lower, 1.0,
                        np.where(gray_norm >= upper, 0.0,
                                 (upper - gray_norm) / delta))
        segmented += i * mask
    segmented = np.round(segmented).astype(np.uint8)
    segmented = median_filter(segmented, size=3)
    segmented = opening(segmented, disk(1))
    segmented = closing(segmented, disk(2))
    return segmented

def calculate_metrics(original, segmented, ground_truth, train_mask):
    data_range = original.max() - original.min()
    psnr_val = psnr(original, segmented, data_range=data_range)
    ssim_val = ssim(original, segmented, data_range=data_range, win_size=3)

    emap_feat = compute_emap(original)
    height, width, n_features = emap_feat.shape
    emap_feat = emap_feat.reshape(-1, n_features)
    lbp_feat = local_binary_pattern(original, P=8, R=1).reshape(-1, 1)
    spectral_feat = original.reshape(-1, 1)
    seg_feat = segmented.reshape(-1, 1)

    X = np.concatenate([emap_feat, spectral_feat, seg_feat, lbp_feat], axis=1)
    y = ground_truth.ravel()

    train_idx = train_mask.ravel() == 1
    test_idx = train_mask.ravel() == 0
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]  # <-- fixed here

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    return psnr_val, ssim_val, oa, kappa

def load_real_data():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/indianpinearray.npy') or not os.path.exists('data/IPgt.npy'):
        print("Downloading Indian Pines dataset...")
        url = "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
        gt_url = "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
        urllib.request.urlretrieve(url, 'data/Indian_pines_corrected.mat')
        urllib.request.urlretrieve(gt_url, 'data/Indian_pines_gt.mat')
        image = loadmat('data/Indian_pines_corrected.mat')['indian_pines_corrected']
        gt = loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']
        np.save('data/indianpinearray.npy', image)
        np.save('data/IPgt.npy', gt)
    else:
        image = np.load('data/indianpinearray.npy')
        gt = np.load('data/IPgt.npy')

    train_mask = np.zeros_like(gt)
    for class_id in np.unique(gt):
        if class_id == 0:
            continue
        pixels = np.argwhere(gt == class_id)
        train_indices = np.random.choice(len(pixels), max(1, int(0.1 * len(pixels))), replace=False)
        train_mask[tuple(pixels[train_indices].T)] = 1

    return image, gt, train_mask

def log_results_to_csv(k, thresholds, psnr_val, ssim_val, oa, kappa, filename="results_log.csv"):
    import csv
    import os
    file_exists = os.path.isfile(filename)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "k", "Thresholds", "PSNR", "SSIM", "OA", "Kappa"])
        writer.writerow([
            timestamp,
            k,
            str(thresholds),
            f"{psnr_val:.2f}",
            f"{ssim_val:.4f}",
            f"{oa:.4f}",
            f"{kappa:.4f}"
        ])

def visualize_results(gray, segmented, gt, k, fitness_curve):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title(f"Gray PCA Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmented, cmap='nipy_spectral')
    plt.title(f"Segmented Output (k={k})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt, cmap='jet')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    if fitness_curve:
        plt.figure(figsize=(6, 4))
        plt.plot(fitness_curve, label='Fuzzy Entropy')
        plt.title(f"Fitness Convergence (k={k})")
        plt.xlabel("Iteration")
        plt.ylabel("Fuzzy Entropy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    image, gt, train_mask = load_real_data()
    pca = PCA(n_components=1)
    gray = pca.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape[:2])
    gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)

    for k in [10, 12, 14]:
        print(f"\nRunning IPSO + Fuzzy Entropy + Local Search for {k} levels...")
        thresholds, _, fitness_curve = ipso_optimize(gray, k_levels=k)
        # Ensure thresholds is a flat list
        if isinstance(thresholds[0], (list, np.ndarray)):
            thresholds = thresholds[0]
        segmented = segment_image(gray, thresholds)
        psnr_val, ssim_val, oa, kappa = calculate_metrics(gray, segmented, gt, train_mask)

        print(f"\nResults for {k} levels:")
        print("Thresholds:", thresholds)
        print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
        print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}")

        log_results_to_csv(k, thresholds, psnr_val, ssim_val, oa, kappa)
        visualize_results(gray, segmented, gt, k, fitness_curve)

if __name__ == "__main__":
    main()
