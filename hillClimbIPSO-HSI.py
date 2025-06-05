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
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import spectral as spy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pathlib import Path
import csv
from scipy.io import loadmat


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


def calculate_metrics(original, segmented, gt, train_mask, classifier_name='MLP'):
    # Calculate data range
    data_range = original.max() - original.min()

    # Now these will work because we imported them properly
    psnr_val = peak_signal_noise_ratio(original, segmented, data_range=data_range)
    ssim_val = structural_similarity(original, segmented, data_range=data_range)

    # Rest of your classification code...
    X = original.reshape(-1, 1)
    y = gt.ravel()

    X_train, y_train = X[train_mask.ravel()], y[train_mask.ravel()]
    X_test, y_test = X[~train_mask.ravel()], y[~train_mask.ravel()]

    if classifier_name == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    else:
        raise ValueError("Unsupported classifier")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    return psnr_val, ssim_val, oa, kappa

def load_dataset(name='IndianPines'):
    os.makedirs('data', exist_ok=True)

    if name == 'IndianPines':
        url = "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
        gt_url = "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
        image_file = 'data/Indian_pines_corrected.mat'
        gt_file = 'data/Indian_pines_gt.mat'
        img_key, gt_key = 'indian_pines_corrected', 'indian_pines_gt'

    elif name == 'Salinas':
        url = "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat"
        gt_url = "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat"
        image_file = 'data/Salinas_corrected.mat'
        gt_file = 'data/Salinas_gt.mat'
        img_key, gt_key = 'salinas_corrected', 'salinas_gt'

    elif name == 'PaviaU':
        url = "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
        gt_url = "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat"
        image_file = 'data/PaviaU.mat'
        gt_file = 'data/PaviaU_gt.mat'
        img_key, gt_key = 'paviaU', 'paviaU_gt'

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    if not os.path.exists(image_file):
        urllib.request.urlretrieve(url, image_file)
    if not os.path.exists(gt_file):
        urllib.request.urlretrieve(gt_url, gt_file)

    image = loadmat(image_file)[img_key]
    gt = loadmat(gt_file)[gt_key]

    train_mask = np.zeros_like(gt)
    for class_id in np.unique(gt):
        if class_id == 0:
            continue
        pixels = np.argwhere(gt == class_id)
        sample_size = max(1, int(0.1 * len(pixels)))
        selected = np.random.choice(len(pixels), sample_size, replace=False)
        train_mask[tuple(pixels[selected].T)] = 1

    return image, gt, train_mask


def log_results_to_csv(k, thresholds, psnr_val, ssim_val, oa, kappa, filename):
    """
    Logs experiment results to a CSV file with headers and automatic file creation

    Args:
        k (int): Number of threshold levels
        thresholds (list): List of threshold values
        psnr_val (float): PSNR value
        ssim_val (float): SSIM value
        oa (float): Overall accuracy
        kappa (float): Kappa coefficient
        filename (str): Output CSV filename
    """
    # Prepare data row
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'k_levels': k,
        'thresholds': '|'.join(map(str, thresholds)),  # Store as pipe-separated string
        'psnr': f"{psnr_val:.4f}",
        'ssim': f"{ssim_val:.4f}",
        'oa': f"{oa:.4f}",
        'kappa': f"{kappa:.4f}"
    }

    # Field names for CSV header
    fieldnames = ['timestamp', 'k_levels', 'thresholds', 'psnr', 'ssim', 'oa', 'kappa']

    # Create parent directories if they don't exist
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write to CSV
    try:
        file_exists = filepath.exists()

        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header only if file doesn't exist
            if not file_exists:
                writer.writeheader()

            writer.writerow(data)

    except Exception as e:
        print(f"Error writing to CSV file {filename}: {str(e)}")
        raise


def visualize_results(original, segmented, gt, k, fitness_curve):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')

    plt.subplot(132)
    plt.imshow(segmented, cmap='nipy_spectral')
    plt.title(f'Segmented (K={k})')

    plt.subplot(133)
    plt.plot(fitness_curve)
    plt.title('Fitness Curve')

    plt.tight_layout()
    plt.show()

def main():
    datasets = ['IndianPines', 'Salinas', 'PaviaU']
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        image, gt, train_mask = load_dataset(name=dataset_name)
        pca = PCA(n_components=1)
        gray = pca.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape[:2])
        gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)

        for k in [10, 12, 14]:
            print(f"\nRunning IPSO + Fuzzy Entropy + Local Search for {k} levels...")
            thresholds, _, fitness_curve = ipso_optimize(gray, k_levels=k)
            segmented = segment_image(gray, thresholds)
            psnr_val, ssim_val, oa, kappa = calculate_metrics(gray, segmented, gt, train_mask, classifier_name='MLP')

            print(f"\nResults for {k} levels:")
            print("Thresholds:", thresholds)
            print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
            print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}")

            log_results_to_csv(k, thresholds, psnr_val, ssim_val, oa, kappa, filename=f"results_{dataset_name}.csv")
            visualize_results(gray, segmented, gt, k, fitness_curve)

if __name__ == "__main__":
    main()