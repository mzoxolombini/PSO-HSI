import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from skimage.util import img_as_ubyte
from skimage.filters.rank import gradient
from skimage.morphology import disk
import warnings

warnings.filterwarnings("ignore")

# PSO Parameters
n_particles = 30
n_iterations = 50
w, c1, c2 = 0.7, 1.5, 1.5  # PSO coefficients


def compute_fuzzy_entropy(image, thresholds):
    """Calculate fuzzy entropy for given thresholds"""
    thresholds = np.sort(thresholds)
    thresholds = np.concatenate(([0], thresholds, [256]))
    total = 0
    for i in range(len(thresholds) - 1):
        mask = (image >= thresholds[i]) & (image < thresholds[i + 1])
        region = image[mask].astype(np.uint8)
        if len(region) == 0:
            continue
        hist, _ = np.histogram(region, bins=256, range=(0, 255), density=True)
        hist += 1e-12  # Avoid log(0)
        total += -np.sum(hist * np.log(hist))
    return total


def compute_spatial_gradient(gray):
    """Compute normalized spatial gradient"""
    gray_ubyte = img_as_ubyte((gray - gray.min()) / (gray.max() - gray.min()))
    spatial_grad = gradient(gray_ubyte, disk(1))
    return spatial_grad / 255.0


def pso_segmentation(image, n_thresholds):
    """
    Pure PSO implementation for multilevel thresholding
    Similar to the reference implementation but without RL/DE components
    """
    # Initialize particles and velocities
    particles = np.random.randint(1, 255, (n_particles, n_thresholds))
    velocities = np.random.randn(n_particles, n_thresholds) * 10

    # Initialize personal and global bests
    pbest = particles.copy()
    pbest_scores = np.array([compute_fuzzy_entropy(image, p) for p in particles])
    gbest = pbest[np.argmax(pbest_scores)]
    gbest_score = np.max(pbest_scores)

    # Optimization loop
    for iter in range(n_iterations):
        for i in range(n_particles):
            # Update velocity and position
            r1, r2 = np.random.rand(n_thresholds), np.random.rand(n_thresholds)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))

            # Apply velocity clamping
            velocities[i] = np.clip(velocities[i], -20, 20)

            # Update position with boundary checking
            particles[i] = np.clip(particles[i] + velocities[i], 1, 254).astype(int)
            particles[i] = np.sort(particles[i])  # Ensure thresholds are ordered

            # Evaluate and update personal best
            current_score = compute_fuzzy_entropy(image, particles[i])
            if current_score > pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = current_score

                # Update global best if needed
                if current_score > gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = current_score

    return np.sort(gbest), gbest_score


def apply_thresholds(image, thresholds):
    """Apply thresholds to create segmented image"""
    thresholds = np.sort(thresholds)
    result = np.zeros_like(image, dtype=np.uint8)
    for i, t in enumerate(thresholds):
        result[image > t] = i + 1
    return result


def load_dataset(name='IndianPines'):
    """Load hyperspectral dataset"""
    from scipy.io import loadmat
    import os
    import urllib.request

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
    return image, gt


def main():
    datasets = ['IndianPines', 'Salinas', 'PaviaU']

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        image, gt = load_dataset(name=dataset_name)

        # Dimensionality reduction using PCA
        h, w, d = image.shape
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(image.reshape(-1, d))
        pc1 = reduced[:, 0].reshape(h, w)  # Use first principal component

        # Normalize to 0-255
        pc1 = ((pc1 - pc1.min()) / (pc1.max() - pc1.min()) * 255).astype(np.uint8)

        results = []
        for k in range(1, 16):  # k from 1 to 15
            print(f"Running PSO for k={k} thresholds...")
            thresholds, score = pso_segmentation(pc1, k)
            segmented = apply_thresholds(pc1, thresholds)

            print(f"Optimal thresholds: {thresholds}")
            print(f"Fuzzy entropy score: {score:.4f}")
            results.append({
                'Dataset': dataset_name,
                'k': k,
                'Thresholds': '|'.join(map(str, thresholds)),
                'Score': score
            })

        # Save results for this dataset
        import pandas as pd
        df = pd.DataFrame(results)
        print(f"\nResults for {dataset_name}:")
        print(df.to_string(index=False))
        df.to_csv(f'{dataset_name}_PSO_results.csv', index=False)


if __name__ == "__main__":
    main()