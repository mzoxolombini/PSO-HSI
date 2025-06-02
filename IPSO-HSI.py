import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, exposure, feature
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numba import jit, prange
import os
import urllib.request
from scipy.io import loadmat
from tqdm import tqdm


# ---------------------- Core IPSO Implementation ----------------------
@jit(nopython=True, cache=True)
def trapezoidal_membership(n, a, b, prev_b, next_a):
    """Optimized trapezoidal membership function with numba acceleration"""
    if prev_b == -1 and next_a == -1:  # First or last region
        if a == 0:  # First region
            return 1.0 if n <= b else max(0.0, (b - n) / (b - a + 1e-10))
        else:  # Last region
            return 0.0 if n <= a else min(1.0, (n - a) / (b - a + 1e-10))
    else:  # Middle regions
        if n <= a:
            return 0.0
        elif n <= b:
            return (n - a) / (b - a + 1e-10)
        elif next_a != -1 and n <= next_a:
            return 1.0
        elif next_a != -1:
            return max(0.0, (next_a - n) / (next_a - b + 1e-10))
        else:
            return 0.0


class IPSO:
    def __init__(self, image, k_levels=10):
        self.image = image
        self.k_levels = k_levels
        self.D = 2 * (k_levels - 1)  # Dimensions (2 params per threshold)
        self.pop_size = 10 * self.D  # Paper uses 10*D
        self.w_initial = 1.2  # From Table 2
        self.c_x = 1.5  # Cognitive coefficient
        self.c_y = 1.5  # Social coefficient
        self.max_iter = 100  # As in paper

        # Initialize swarms for each dimension
        self.swarms = [self._init_swarm() for _ in range(self.D)]
        self.CP = np.zeros(self.D)  # Context Parameter
        self.best_fitness_history = []

    def _init_swarm(self):
        """Initialize a single swarm"""
        return {
            'positions': np.random.uniform(0, 255, (self.pop_size, 1)),
            'velocities': np.zeros((self.pop_size, 1)),
            'pb_positions': np.zeros((self.pop_size, 1)),
            'pb_fitness': np.full(self.pop_size, -np.inf),
            'gb_position': np.zeros(1),
            'gb_fitness': -np.inf
        }

    def evaluate_fitness(self, particle_position, dim):
        """Evaluate fitness using context parameter"""
        temp_CP = self.CP.copy()
        temp_CP[dim] = particle_position[0]
        return self.calculate_fuzzy_entropy(temp_CP)

    @jit(nopython=True, parallel=True)
    def calculate_fuzzy_entropy(self, thresholds):
        """Parallelized fuzzy entropy calculation"""
        hist, _ = np.histogram(self.image, bins=256, range=(0, 255))
        hist = hist / hist.sum() + 1e-10
        total_entropy = 0.0

        # Sort and add boundary thresholds
        thresholds = np.sort(thresholds)
        thresholds = np.concatenate(([0], thresholds, [255]))

        for k in prange(1, len(thresholds)):
            a = thresholds[k - 1]
            b = thresholds[k]
            prev_b = thresholds[k - 2] if k > 1 else -1
            next_a = thresholds[k + 1] if k < len(thresholds) - 1 else -1

            # Calculate membership values
            mu = np.zeros(256)
            for n in range(256):
                mu[n] = trapezoidal_membership(n, a, b, prev_b, next_a)

            R_k = np.sum(hist * mu)
            if R_k > 1e-10:
                p = (hist * mu) / R_k
                p = np.clip(p, 1e-10, 1)
                H_k = -np.sum(p * np.log(p))
                total_entropy += H_k

        return total_entropy

    def optimize(self):
        """Main optimization loop with improvements from paper"""
        # Initialize personal bests
        for dim in range(self.D):
            swarm = self.swarms[dim]
            for i in range(self.pop_size):
                fitness = self.evaluate_fitness(swarm['positions'][i], dim)
                swarm['pb_fitness'][i] = fitness
                swarm['pb_positions'][i] = swarm['positions'][i].copy()
                if fitness > swarm['gb_fitness']:
                    swarm['gb_fitness'] = fitness
                    swarm['gb_position'] = swarm['positions'][i].copy()
            self.CP[dim] = swarm['gb_position'][0]

        # Optimization with adaptive inertia
        for iter in tqdm(range(self.max_iter), desc="IPSO Optimization"):
            w = self.w_initial - (self.w_initial - 0.4) * iter / self.max_iter

            for dim in range(self.D):
                swarm = self.swarms[dim]

                # Update particles
                for i in range(self.pop_size):
                    # Velocity update with tournament selection
                    if np.random.rand() > 0.5:  # Paper's selection probability
                        # Learn from own pb
                        cognitive = self.c_x * np.random.rand() * (swarm['pb_positions'][i] - swarm['positions'][i])
                    else:
                        # Tournament selection for another particle's pb
                        idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
                        best_idx = idx1 if swarm['pb_fitness'][idx1] > swarm['pb_fitness'][idx2] else idx2
                        cognitive = self.c_x * np.random.rand() * (
                                    swarm['pb_positions'][best_idx] - swarm['positions'][i])

                    social = self.c_y * np.random.rand() * (swarm['gb_position'] - swarm['positions'][i])
                    swarm['velocities'][i] = w * swarm['velocities'][i] + cognitive + social

                    # Position update with clamping
                    swarm['positions'][i] = np.clip(swarm['positions'][i] + swarm['velocities'][i], 0, 255)

                    # Evaluate and update
                    fitness = self.evaluate_fitness(swarm['positions'][i], dim)
                    if fitness > swarm['pb_fitness'][i]:
                        swarm['pb_fitness'][i] = fitness
                        swarm['pb_positions'][i] = swarm['positions'][i].copy()
                        if fitness > swarm['gb_fitness']:
                            swarm['gb_fitness'] = fitness
                            swarm['gb_position'] = swarm['positions'][i].copy()

                # Update context parameter
                self.CP[dim] = swarm['gb_position'][0]

                # Best-worst particle replacement (Sec 4.3)
                if iter > 10 and np.std(swarm['pb_fitness']) < 1e-5:  # Stagnation detection
                    sorted_idx = np.argsort(swarm['pb_fitness'])
                    replace_count = self.pop_size // 2
                    worst_idx = sorted_idx[:replace_count]
                    best_idx = sorted_idx[-replace_count:]
                    swarm['positions'][worst_idx] = swarm['positions'][best_idx]
                    swarm['velocities'][worst_idx] = 0  # Reset velocity

            self.best_fitness_history.append(max(s['gb_fitness'] for s in self.swarms))

        # Calculate final thresholds (Eq. 7 in paper)
        thresholds = [(self.CP[2 * i] + self.CP[2 * i + 1]) / 2 for i in range(self.k_levels - 1)]
        return [0] + sorted(thresholds) + [255]


# ---------------------- Image Processing ----------------------
def preprocess_hyperspectral(image):
    """Dimensionality reduction as described in paper (Sec 5.1)"""
    orig_shape = image.shape
    data = image.reshape(-1, orig_shape[-1])

    # Remove noisy bands (like water absorption bands)
    valid_bands = np.std(data, axis=0) > 1e-5
    data = data[:, valid_bands]

    # PCA keeping 95% variance
    pca = PCA(n_components=0.95)
    pc = pca.fit_transform(data)
    return pc.reshape(orig_shape[0], orig_shape[1], -1)


def create_emap(pc_image, attributes=['area', 'std']):
    """Create EMAP features following paper's methodology (Sec 5.2)"""
    from skimage.morphology import disk, area_opening

    profiles = []
    for i in range(pc_image.shape[-1]):
        band = exposure.rescale_intensity(pc_image[..., i], out_range=(0, 255)).astype(np.uint8)

        for attr in attributes:
            for size in [3, 5, 7]:  # Multiple scales
                if attr == 'area':
                    profile = area_opening(band, area_threshold=size)
                elif attr == 'std':
                    profile = filters.rank.gradient(band, disk(size))
                profiles.append(profile)

    return np.stack(profiles, axis=-1)


def segment_image(gray, thresholds):
    """Improved multi-level thresholding with postprocessing"""
    segmented = np.zeros_like(gray, dtype=np.uint8)
    for i in range(len(thresholds) - 1):
        mask = (gray >= thresholds[i]) & (gray < thresholds[i + 1])
        segmented[mask] = i

    # Post-processing (Sec 6.1)
    segmented = median_filter(segmented, size=3)
    for i in np.unique(segmented):
        segmented = morphology.opening(segmented == i, morphology.disk(1))
        segmented = morphology.closing(segmented == i, morphology.disk(2))
    return segmented


# ---------------------- Classification ----------------------
def composite_kernel_svm(X_spatial, X_spectral, y, train_mask):
    """Implements the paper's composite kernel approach (Sec 3.2)"""
    # Normalize features
    scaler = StandardScaler()
    X_spatial = scaler.fit_transform(X_spatial.reshape(-1, X_spatial.shape[-1]))
    X_spectral = scaler.transform(X_spectral.reshape(-1, X_spectral.shape[-1]))

    # Kernel composition (70% spatial + 30% spectral)
    kernel = 0.7 * RBF(length_scale=1.0) + 0.3 * RBF(length_scale=1.0)

    # SVM with class weights
    svm = SVC(kernel=kernel, class_weight='balanced', C=10, gamma='scale')
    svm.fit(np.hstack([X_spatial[train_mask.ravel() == 1],
                       X_spectral[train_mask.ravel() == 1]]),
            y[train_mask.ravel() == 1])
    return svm


def calculate_metrics(svm, X_spatial, X_spectral, y, train_mask):
    """Calculate all evaluation metrics from paper"""
    X_test = np.hstack([X_spatial[train_mask.ravel() == 0],
                        X_spectral[train_mask.ravel() == 0]])
    y_test = y[train_mask.ravel() == 0]
    y_pred = svm.predict(X_test)

    # OA (Overall Accuracy)
    oa = accuracy_score(y_test, y_pred)

    # Kappa Index
    kappa = cohen_kappa_score(y_test, y_pred)

    # MA (Mean Accuracy)
    classes = np.unique(y_test[y_test != 0])  # Exclude background
    ma = np.mean([accuracy_score(y_test[y_test == c], y_pred[y_test == c]) for c in classes])

    # IoU (Intersection over Union)
    iou = np.mean([np.sum((y_test == c) & (y_pred == c)) /
                   np.sum((y_test == c) | (y_pred == c)) for c in classes])

    return oa, kappa, ma, iou


# ---------------------- Main Pipeline ----------------------
def load_indian_pines():
    """Load and preprocess Indian Pines dataset"""
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/Indian_pines_corrected.mat'):
        urllib.request.urlretrieve(
            "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            "data/Indian_pines_corrected.mat")
        urllib.request.urlretrieve(
            "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
            "data/Indian_pines_gt.mat")

    image = loadmat('data/Indian_pines_corrected.mat')['indian_pines_corrected']
    gt = loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']

    # Create training mask (10% per class)
    train_mask = np.zeros_like(gt)
    for c in np.unique(gt):
        if c == 0: continue
        idx = np.argwhere(gt == c)
        np.random.shuffle(idx)
        train_mask[tuple(idx[:int(0.1 * len(idx))].T)] = 1

    return image, gt, train_mask


def main():
    # Load and preprocess
    image, gt, train_mask = load_indian_pines()
    pc_image = preprocess_hyperspectral(image)

    # Use first PC for segmentation (as in paper)
    gray = exposure.rescale_intensity(pc_image[..., 0], out_range=(0, 255)).astype(np.uint8)

    results = []
    for k in [10, 12, 14]:
        print(f"\nProcessing k={k} levels...")
        start_time = time.time()

        # 1. IPSO Segmentation
        ipso = IPSO(gray, k_levels=k)
        thresholds = ipso.optimize()
        segmented = segment_image(gray, thresholds)

        # 2. Feature Extraction
        emap = create_emap(pc_image)  # Spatial features
        spectral_feat = np.stack([segmented], axis=-1)  # Spectral features

        # 3. Classification
        svm = composite_kernel_svm(emap, spectral_feat, gt, train_mask)
        oa, kappa, ma, iou = calculate_metrics(svm, emap, spectral_feat, gt, train_mask)

        # 4. Evaluation
        psnr_val = psnr(gray, segmented, data_range=255)
        ssim_val = ssim(gray, segmented, data_range=255)

        results.append({
            'k': k,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'oa': oa,
            'kappa': kappa,
            'ma': ma,
            'iou': iou,
            'time': time.time() - start_time
        })

        # Visualization
        plt.figure(figsize=(15, 5))
        plt.subplot(131);
        plt.imshow(gray, cmap='gray');
        plt.title('Original')
        plt.subplot(132);
        plt.imshow(segmented);
        plt.title('Segmented')
        plt.subplot(133);
        plt.imshow(gt);
        plt.title('Ground Truth')
        plt.show()

    # Print results
    print("\nFinal Results:")
    print("k | PSNR | SSIM | OA | Kappa | MA | IoU | Time (s)")
    for r in results:
        print(f"{r['k']} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['oa']:.4f} | "
              f"{r['kappa']:.4f} | {r['ma']:.4f} | {r['iou']:.4f} | {r['time']:.1f}")


if __name__ == "__main__":
    main()