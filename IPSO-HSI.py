import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import disk, opening, closing
from skimage.filters import rank
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.ndimage import median_filter
import os
import urllib.request
from scipy.io import loadmat

class IPSO:
    def __init__(self, image, k_levels=10, population_size=20, max_iter=30,
                 w_initial=0.7, c_x=1.7, c_y=1.7, replace_ratio=0.3):
        self.image = image
        self.k_levels = k_levels
        self.pop_size = population_size
        self.max_iter = max_iter
        self.w_initial = w_initial
        self.c_x = c_x
        self.c_y = c_y
        self.replace_ratio = replace_ratio

        self.pca = PCA(n_components=1)
        self.gray = self.pca.fit_transform(image.reshape(-1, image.shape[-1]))
        self.gray = self.gray.reshape(image.shape[0], image.shape[1])
        self.gray = ((self.gray - self.gray.min()) /
                     (self.gray.max() - self.gray.min()) * 255).astype(np.uint8)

        self.spatial_feat = filters.sobel(self.gray)
        self.D = 2 * (k_levels - 1)
        self.eps = 1e-10

        self.swarms = [{
            'positions': np.random.uniform(0, 255, (population_size, 1)),
            'velocities': np.zeros((population_size, 1)),
            'pb_positions': np.zeros((population_size, 1)),
            'pb_fitness': np.full(population_size, -np.inf),
            'gb_position': np.zeros(1),
            'gb_fitness': -np.inf,
            'stagnation_count': 0
        } for _ in range(self.D)]

        self.CP = np.zeros(self.D)
        self.best_fitness_history = []

    def trapezoidal_membership(self, n, a, b):
        n = np.clip(n, 0, 255)
        if b - a < 1e-6:
            return np.where(n <= a, 1.0, 0.0)
        return np.where(n <= a, 1.0, np.where(n <= b, (b - n) / (b - a), 0.0))

    def calculate_fuzzy_entropy(self, thresholds):
        hist, _ = np.histogram(self.gray, bins=256, range=(0, 255))
        hist = np.cumsum(hist) / np.sum(hist)

        total_entropy = 0.0
        thresholds = [0] + sorted(thresholds) + [255]
        x = np.arange(256)

        for k in range(1, self.k_levels + 1):
            if k == 1:
                a, b = 0, thresholds[1]
                mu = self.trapezoidal_membership(x, a, b)
            elif k == self.k_levels:
                a, b = thresholds[-2], 255
                mu = self.trapezoidal_membership(x, a, b)
            else:
                a_prev, b_prev = thresholds[2 * k - 2], thresholds[2 * k - 1]
                a, b = thresholds[2 * k], thresholds[2 * k + 1]
                delta1 = max(b_prev - a_prev, self.eps)
                delta2 = max(b - a, self.eps)

                mu = np.where(x <= a_prev, 0,
                              np.where(x <= b_prev,
                                       (x - a_prev) / delta1,
                                       np.where(x <= a, 1,
                                                np.where(x <= b,
                                                         (b - x) / delta2,
                                                         0))))

            if np.isnan(mu).any():
                continue

            R_k = np.sum(hist * mu)
            if R_k > self.eps:
                terms = (hist * mu) / R_k
                terms = np.clip(terms, self.eps, 1)
                H_k = -np.sum(terms * np.log(terms))
                total_entropy += H_k

        return total_entropy

    def evaluate_fitness(self, particle_positions, dim):
        temp_CP = self.CP.copy()
        temp_CP[dim] = float(particle_positions[0])
        return self.calculate_fuzzy_entropy(temp_CP)

    def update_velocity_position(self, swarm_idx):
        swarm = self.swarms[swarm_idx]
        w = self.w_initial - (self.w_initial / self.max_iter) * self.iteration

        for i in range(self.pop_size):
            f = i if np.random.rand() > 0.5 else \
                np.random.choice(self.pop_size, 2, replace=False)[
                    np.argmax(swarm['pb_fitness'][np.random.choice(self.pop_size, 2)])]

            r1, r2 = np.random.rand(), np.random.rand()
            cognitive = self.c_x * r1 * (swarm['pb_positions'][f] - swarm['positions'][i])
            social = self.c_y * r2 * (swarm['gb_position'] - swarm['positions'][i])
            swarm['velocities'][i] = np.clip(
                w * swarm['velocities'][i] + cognitive + social, -10, 10)

            swarm['positions'][i] = np.clip(
                swarm['positions'][i] + swarm['velocities'][i], 0, 255)

    def replace_worst_particles(self, swarm_idx):
        swarm = self.swarms[swarm_idx]
        num_replace = int(self.pop_size * self.replace_ratio)
        idx = np.argpartition(swarm['pb_fitness'], num_replace)

        swarm['positions'][idx[:num_replace]] = swarm['positions'][idx[-num_replace:]].copy()
        swarm['velocities'][idx[:num_replace]] = 0
        swarm['pb_positions'][idx[:num_replace]] = swarm['pb_positions'][idx[-num_replace:]].copy()
        swarm['pb_fitness'][idx[:num_replace]] = swarm['pb_fitness'][idx[-num_replace:]].copy()

    def optimize(self):
        start_time = time.time()

        for dim in range(self.D):
            swarm = self.swarms[dim]
            for i in range(self.pop_size):
                swarm['pb_fitness'][i] = self.evaluate_fitness(swarm['positions'][i], dim)
                swarm['pb_positions'][i] = swarm['positions'][i].copy()
                if swarm['pb_fitness'][i] > swarm['gb_fitness']:
                    swarm['gb_fitness'] = swarm['pb_fitness'][i]
                    swarm['gb_position'] = swarm['positions'][i].copy()
            self.CP[dim] = float(swarm['gb_position'][0])

        for self.iteration in range(self.max_iter):
            for dim, swarm in enumerate(self.swarms):
                if swarm['stagnation_count'] >= 5:
                    self.replace_worst_particles(dim)
                    swarm['stagnation_count'] = 0

                self.update_velocity_position(dim)

                for i in range(self.pop_size):
                    fitness = self.evaluate_fitness(swarm['positions'][i], dim)
                    if fitness > swarm['pb_fitness'][i]:
                        swarm['pb_fitness'][i] = fitness
                        swarm['pb_positions'][i] = swarm['positions'][i].copy()
                        if fitness > swarm['gb_fitness']:
                            swarm['gb_fitness'] = fitness
                            swarm['gb_position'] = swarm['positions'][i].copy()

                self.CP[dim] = float(swarm['gb_position'][0])

            self.best_fitness_history.append(max(s['gb_fitness'] for s in self.swarms))
            if (self.iteration + 1) % 10 == 0:
                print(f"Iter {self.iteration + 1}/{self.max_iter}, Best: {self.best_fitness_history[-1]:.4f}")

        thresholds = sorted((self.CP[2 * i] + self.CP[2 * i + 1]) / 2 for i in range(self.k_levels - 1))
        return [0] + thresholds + [255], time.time() - start_time, self.best_fitness_history


def segment_image(gray, thresholds):
    segmented = np.zeros_like(gray, dtype=np.float32)
    gray_norm = gray.astype(np.float32) / 255.0

    for i in range(len(thresholds)-1):
        lower = thresholds[i] / 255.0
        upper = thresholds[i+1] / 255.0
        delta = max(upper - lower, 1e-10)
        mask = np.where(gray_norm <= lower, 1.0,
                       np.where(gray_norm >= upper, 0.0,
                              (upper - gray_norm)/delta))
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

    spectral_feat = original.reshape(-1, 1)
    spatial_feat = segmented.reshape(-1, 1)
    X = np.concatenate([spectral_feat, spatial_feat], axis=1)
    y = ground_truth.ravel()

    X_train, y_train = X[train_mask.ravel() == 1], y[train_mask.ravel() == 1]
    X_test, y_test = X[train_mask.ravel() == 0], y[train_mask.ravel() == 0]

    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10, gamma='auto', class_weight='balanced')
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return psnr_val, ssim_val, accuracy_score(y_test, y_pred), cohen_kappa_score(y_test, y_pred)


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
        if class_id == 0: continue
        pixels = np.argwhere(gt == class_id)
        train_indices = np.random.choice(len(pixels), max(1, int(0.1 * len(pixels))), replace=False)
        train_mask[tuple(pixels[train_indices].T)] = 1

    return image, gt, train_mask


def main():
    image, gt, train_mask = load_real_data()

    for k in [10, 12, 14]:
        print(f"\nRunning IPSO for {k} levels...")
        ipso = IPSO(image, k_levels=k)
        thresholds, time_elapsed, _ = ipso.optimize()

        segmented = segment_image(ipso.gray, thresholds)
        psnr_val, ssim_val, oa, kappa = calculate_metrics(ipso.gray, segmented, gt, train_mask)

        print(f"\nResults for {k} levels:")
        threshold_pairs = [(int(round(thresholds[i])), int(round(thresholds[i + 1])))
                           for i in range(len(thresholds) - 1)]
        print("Thresholds:", ', '.join(f"({a}, {b})" for a, b in threshold_pairs))
        print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
        print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}")
        print(f"Time: {time_elapsed:.2f}s")

        plt.figure(figsize=(15, 10))
        plt.subplot(131); plt.imshow(ipso.gray, cmap='gray'); plt.title('PCA Component')
        plt.subplot(132); plt.imshow(segmented); plt.title('Segmentation')
        plt.subplot(133); plt.imshow(gt); plt.title('Ground Truth')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
