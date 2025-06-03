import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, exposure, feature, segmentation
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter, gaussian_filter
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numba import jit, prange
import os
import urllib.request
from scipy.io import loadmat
from tqdm import tqdm
from skimage.morphology import disk, square, dilation, erosion
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from skimage import color, exposure



# ---------------------- Improved IPSO Implementation ----------------------
@jit(nopython=True)
def trapezoidal_membership(n, a, b, prev_b, next_a):
    """More precise membership function"""
    if prev_b == -1:  # First segment
        return max(0.0, min(1.0, (b - n)/(b - a + 1e-10)))
    elif next_a == -1:  # Last segment
        return max(0.0, min(1.0, (n - a)/(b - a + 1e-10)))
    else:  # Middle segments
        if n <= a: return 0.0
        elif n >= b: return 0.0
        else: return 1.0


@jit(nopython=True)
def calculate_color_fuzzy_entropy(image_lab_flat, thresholds):
    """Compute fuzzy entropy over L*, a*, b* color histograms."""
    total_entropy = 0.0

    for ch in range(3):  # L*, a*, b* channels
        hist = np.zeros(256)
        for i in range(image_lab_flat.shape[0]):
            hist[int(image_lab_flat[i, ch])] += 1
        hist = hist / (hist.sum() + 1e-10)

        thresholds_sorted = np.sort(thresholds)
        thresholds_full = np.concatenate((np.array([0.0]), thresholds_sorted, np.array([255.0])))

        for k in range(1, len(thresholds_full)):
            a = thresholds_full[k - 1]
            b = thresholds_full[k]
            prev_b = thresholds_full[k - 2] if k > 1 else -1.0
            next_a = thresholds_full[k + 1] if k < len(thresholds_full) - 1 else -1.0

            R_k = 0.0
            for n in range(256):
                mu = trapezoidal_membership(float(n), a, b, prev_b, next_a)
                R_k += hist[n] * mu

            H_k = 0.0
            if R_k > 1e-10:
                for n in range(256):
                    mu = trapezoidal_membership(float(n), a, b, prev_b, next_a)
                    p = (hist[n] * mu) / R_k
                    if p > 1e-10:
                        H_k -= p * np.log(p)

            total_entropy += H_k

    return total_entropy

class ImprovedIPSO:
    def __init__(self, image, k_levels=10):
        self.image = image
        self.k_levels = k_levels
        self.D = 2 * (k_levels - 1)  # a and b for each threshold
        self.pop_size = 50 * self.D  # Increased population
        self.w_initial = 0.9
        self.w_final = 0.4
        self.c_x = 2.0
        self.c_y = 2.0
        self.max_iter = 100  # Increased iterations
        self.stagnation_threshold = 10

        # Initialize multiple swarms for each dimension
        self.swarms = [self._init_swarm() for _ in range(self.D)]
        self.CP = np.zeros(self.D)  # Context parameter
        self.best_fitness_history = []
        self.stagnation_count = np.zeros(self.D)

    def _init_swarm(self):
        """Initialize swarm with proper velocity limits"""
        positions = np.random.uniform(0, 255, (self.pop_size, 1))
        velocities = np.random.uniform(-10, 10, (self.pop_size, 1))
        return {
            'positions': positions,
            'velocities': velocities,
            'pb_positions': positions.copy(),
            'pb_fitness': np.full(self.pop_size, -np.inf),
            'gb_position': np.zeros(1),
            'gb_fitness': -np.inf
        }

    def evaluate_fitness(self, particle_position, dim):
        """Evaluate fitness using context parameter"""
        temp_CP = self.CP.copy()
        temp_CP[dim] = particle_position[0]
        return self.calculate_fuzzy_entropy(temp_CP)

    def calculate_fuzzy_entropy(self, thresholds):
        return calculate_color_fuzzy_entropy(
            self.image.reshape(-1, 3).astype(np.float64),
            np.array(thresholds, dtype=np.float64)
        )

    def optimize(self):
        """Improved optimization with better stagnation handling"""
        # Initialize personal and global bests
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

        # Main optimization loop
        for iter in tqdm(range(self.max_iter), desc="Improved IPSO"):
            w = self.w_initial - (self.w_initial - self.w_final) * (iter / self.max_iter)

            for dim in range(self.D):
                swarm = self.swarms[dim]
                prev_best = swarm['gb_fitness']

                for i in range(self.pop_size):
                    # Cognitive component - choose between personal best or tournament
                    if np.random.rand() > 0.7:  # 30% chance to use tournament
                        idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
                        best_idx = idx1 if swarm['pb_fitness'][idx1] > swarm['pb_fitness'][idx2] else idx2
                        cognitive = self.c_x * np.random.rand() * (
                                    swarm['pb_positions'][best_idx] - swarm['positions'][i])
                    else:
                        cognitive = self.c_x * np.random.rand() * (swarm['pb_positions'][i] - swarm['positions'][i])

                    # Social component
                    social = self.c_y * np.random.rand() * (swarm['gb_position'] - swarm['positions'][i])

                    # Update velocity with clamping
                    swarm['velocities'][i] = w * swarm['velocities'][i] + cognitive + social
                    swarm['velocities'][i] = np.clip(swarm['velocities'][i], -25, 25)

                    # Update position
                    swarm['positions'][i] = np.clip(swarm['positions'][i] + swarm['velocities'][i], 0, 255)

                    # Evaluate and update personal best
                    fitness = self.evaluate_fitness(swarm['positions'][i], dim)
                    if fitness > swarm['pb_fitness'][i]:
                        swarm['pb_fitness'][i] = fitness
                        swarm['pb_positions'][i] = swarm['positions'][i].copy()
                        if fitness > swarm['gb_fitness']:
                            swarm['gb_fitness'] = fitness
                            swarm['gb_position'] = swarm['positions'][i].copy()

                # Update context parameter
                self.CP[dim] = swarm['gb_position'][0]

                # Check for stagnation
                if abs(swarm['gb_fitness'] - prev_best) < 1e-5:
                    self.stagnation_count[dim] += 1
                    if self.stagnation_count[dim] > self.stagnation_threshold:
                        self._handle_stagnation(dim)
                else:
                    self.stagnation_count[dim] = 0

            self.best_fitness_history.append(max(s['gb_fitness'] for s in self.swarms))

        # Calculate final thresholds
        thresholds = [(self.CP[2 * i] + self.CP[2 * i + 1]) / 2 for i in range(self.k_levels - 1)]
        return [0] + sorted(thresholds) + [255]

    def _handle_stagnation(self, dim):
        """Replace worst particles with mutated versions of best particles"""
        swarm = self.swarms[dim]
        sorted_idx = np.argsort(swarm['pb_fitness'])
        replace_count = self.pop_size // 3  # Replace 1/3 of population

        worst_idx = sorted_idx[:replace_count]
        best_idx = sorted_idx[-replace_count:]

        # Create mutated versions of best particles
        for i, j in zip(worst_idx, best_idx):
            swarm['positions'][i] = np.clip(swarm['pb_positions'][j] + np.random.normal(0, 10), 0, 255)
            swarm['velocities'][i] = np.random.uniform(-10, 10, 1)
            swarm['pb_positions'][i] = swarm['positions'][i].copy()
            swarm['pb_fitness'][i] = self.evaluate_fitness(swarm['positions'][i], dim)

        self.stagnation_count[dim] = 0


# ---------------------- Enhanced Image Processing ----------------------
def preprocess_hyperspectral(image):
    """Improved preprocessing with band selection and noise removal"""
    orig_shape = image.shape
    data = image.reshape(-1, orig_shape[-1])

    # Remove noisy bands (like water absorption bands)
    band_std = np.std(data, axis=0)
    valid_bands = (band_std > np.percentile(band_std, 5)) & (band_std < np.percentile(band_std, 95))
    data = data[:, valid_bands]

    # Whitening transformation
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-10)

    # PCA keeping 95% variance
    pca = PCA(n_components=0.95)
    pc = pca.fit_transform(data)

    return pc.reshape(orig_shape[0], orig_shape[1], -1)

def convert_to_lab(image):
    """Convert PCA-reduced HSI to RGB then to Lab color space."""
    # Use first 3 PCs as RGB approximation
    rgb_approx = image[..., :3]
    rgb_norm = (rgb_approx - rgb_approx.min()) / (rgb_approx.max() - rgb_approx.min())
    rgb_norm = np.clip(rgb_norm, 0, 1)

    # Convert to Lab
    lab_image = color.rgb2lab(rgb_norm)

    # Rescale each channel to [0, 255]
    lab_scaled = np.zeros_like(lab_image)
    lab_scaled[..., 0] = exposure.rescale_intensity(lab_image[..., 0], out_range=(0, 255))  # L*
    lab_scaled[..., 1] = exposure.rescale_intensity(lab_image[..., 1], out_range=(0, 255))  # a*
    lab_scaled[..., 2] = exposure.rescale_intensity(lab_image[..., 2], out_range=(0, 255))  # b*

    return lab_scaled.astype(np.uint8)


def create_emap(pc_image):
    """Create Extended Multi-Attribute Profiles as in the paper"""
    profiles = []
    for i in range(min(5, pc_image.shape[-1])):  # Use first 5 PCs
        band = exposure.rescale_intensity(pc_image[..., i], out_range=(0, 255))

        # Attribute profiles with multiple structuring elements
        for radius in [3, 5, 7]:
            profiles.extend([
                morphology.opening(band, disk(radius)),
                morphology.closing(band, disk(radius)),
                morphology.dilation(band, disk(radius)),
                morphology.erosion(band, disk(radius))
            ])

        # Add texture features
        profiles.append(feature.local_binary_pattern(band, 8, 1.0))
        profiles.append(filters.gaussian(band, sigma=1))
        profiles.append(filters.gaussian(band, sigma=3))

    return np.stack(profiles, axis=-1)



def segment_image(gray, thresholds):
    """Improved segmentation with proper post-processing"""
    # Create segmented image
    segmented = np.zeros_like(gray, dtype=np.uint8)
    for i in range(len(thresholds) - 1):
        mask = (gray >= thresholds[i]) & (gray < thresholds[i + 1])
        segmented[mask] = i

    # Enhanced post-processing
    segmented = median_filter(segmented, size=3)
    return segmented


def morphological_refinement(segmented):
    """Apply morphological operations as in paper"""
    refined = np.zeros_like(segmented)
    for class_id in np.unique(segmented):
        if class_id == 0:  # Skip background
            continue

        class_mask = segmented == class_id

        # Remove small objects
        cleaned = morphology.remove_small_objects(class_mask, min_size=50)

        # Fill holes
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=25)

        # Smooth boundaries
        cleaned = gaussian_filter(cleaned.astype(float), sigma=1) > 0.5

        refined[cleaned] = class_id

    return refined

# ---------------------- Enhanced Classification ----------------------
def create_composite_kernel(X_spatial, X_spectral):
    """Create composite kernel as described in paper"""
    # Spatial kernel (RBF)
    spatial_kernel = RBF(length_scale=1.0)

    # Spectral kernel (RBF)
    spectral_kernel = RBF(length_scale=1.0)

    # Combine with cross-information kernel
    return ConstantKernel(1.0) * spatial_kernel + ConstantKernel(1.0) * spectral_kernel


from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def composite_kernel_svm(X_spatial, X_spectral, y, train_mask):
    # Reshape features
    X_spatial_flat = X_spatial.reshape(-1, X_spatial.shape[-1])
    X_spectral_flat = X_spectral.reshape(-1, X_spectral.shape[-1])

    # Create composite kernel as in paper
    spatial_kernel = RBF(length_scale=1.0)
    spectral_kernel = RBF(length_scale=1.0)
    composite_kernel = ConstantKernel(1.0) * spatial_kernel + ConstantKernel(1.0) * spectral_kernel

    # Train SVM with composite kernel
    svm = SVC(kernel=composite_kernel, C=1.0, gamma='scale',
              class_weight='balanced')

    # Get training samples
    train_idx = train_mask.flatten() == 1
    X_train = np.hstack([X_spatial_flat[train_idx], X_spectral_flat[train_idx]])
    y_train = y.flatten()[train_idx]

    svm.fit(X_train, y_train)
    return svm



# ---------------------- Main Pipeline ----------------------
def load_indian_pines():
    """Load and preprocess Indian Pines dataset with proper training samples"""
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

    # Create training mask (10% per class as in paper)
    train_mask = np.zeros_like(gt)
    for c in np.unique(gt):
        if c == 0: continue
        idx = np.argwhere(gt == c)
        np.random.shuffle(idx)
        train_samples = idx[:max(1, int(0.1 * len(idx)))]  # At least 1 sample per class
        train_mask[tuple(train_samples.T)] = 1

    return image, gt, train_mask

def calculate_metrics(model, X_spatial, X_spectral, gt, train_mask):
    """Calculate classification metrics: OA, Kappa, MA, and IoU"""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

    height, width = gt.shape
    X_spatial_flat = X_spatial.reshape(-1, X_spatial.shape[-1])
    X_spectral_flat = X_spectral.reshape(-1, X_spectral.shape[-1])
    X = np.hstack([X_spatial_flat, X_spectral_flat])

    # Predict all pixels
    predictions = model.predict(X).reshape(height, width)

    # Mask out background
    mask = gt > 0
    gt_masked = gt[mask]
    pred_masked = predictions[mask]

    # Overall Accuracy
    oa = accuracy_score(gt_masked, pred_masked)

    # Kappa
    kappa = cohen_kappa_score(gt_masked, pred_masked)

    # Confusion matrix
    cm = confusion_matrix(gt_masked, pred_masked, labels=np.unique(gt_masked))

    # Mean Accuracy (MA)
    with np.errstate(divide='ignore', invalid='ignore'):
        ma = np.nanmean(np.diag(cm) / np.maximum(1, cm.sum(axis=1)))

    # Intersection over Union (IoU)
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = np.nanmean(intersection / np.maximum(union, 1))

    return oa, kappa, ma, iou

def main():
    # Load and preprocess
    image, gt, train_mask = load_indian_pines()
    pc_image = preprocess_hyperspectral(image)

    results = []

    for k in [10, 12, 14]:
        print(f"\nProcessing k={k} levels...")
        start_time = time.time()

        # Convert PCA image to Lab
        lab_image = convert_to_lab(pc_image)
        gray = lab_image[..., 0]  # Use L* channel for segmentation

        # 1. Improved IPSO Segmentation
        ipso = ImprovedIPSO(lab_image, k_levels=k)
        thresholds = ipso.optimize()
        segmented = segment_image(gray, thresholds)

        # Apply morphological refinement
        segmented = morphological_refinement(segmented)

        # 2. Enhanced Feature Extraction
        emap = create_emap(pc_image)  # Spatial features
        spectral_feat = np.stack([segmented], axis=-1)  # Spectral features

        # 3. Improved Classification
        svm = composite_kernel_svm(emap, spectral_feat, gt, train_mask)
        oa, kappa, ma, iou = calculate_metrics(svm, emap, spectral_feat, gt, train_mask)

        # 4. Evaluation
        psnr_val = psnr(gray, segmented, data_range=255)
        ssim_val = ssim(gray, segmented, win_size=7, channel_axis=None, data_range=255)

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
        plt.subplot(131)
        plt.imshow(gray, cmap='gray')
        plt.title('L* Channel')
        plt.subplot(132)
        plt.imshow(segmented)
        plt.title(f'Segmented (k={k})')
        plt.subplot(133)
        plt.imshow(gt)
        plt.title('Ground Truth')
        plt.show()

    # Print results
    print("\nFinal Results:")
    print("k | PSNR | SSIM | OA | Kappa | MA | IoU | Time (s)")
    for r in results:
        print(f"{r['k']} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['oa']:.4f} | "
              f"{r['kappa']:.4f} | {r['ma']:.4f} | {r['iou']:.4f} | {r['time']:.1f}")


if __name__ == "__main__":
    from sklearn.utils.class_weight import compute_class_weight

    main()