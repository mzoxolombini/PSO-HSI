import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import warnings
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    try:
        from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
    except ImportError:
        def graycomatrix(*args, **kwargs):
            raise ImportError("GLCM features not available")

        def graycoprops(*args, **kwargs):
            raise ImportError("GLCM features not available")

warnings.filterwarnings("ignore")

n_particles = 30
n_iterations = 100
w_initial = 0.9
c1_initial = 2.5
c2_initial = 0.5

def compute_fuzzy_entropy(image, thresholds):
    thresholds = np.sort(thresholds)
    thresholds = np.concatenate(([0], thresholds, [256]))
    total = 0
    for i in range(len(thresholds) - 1):
        mask = (image >= thresholds[i]) & (image < thresholds[i + 1])
        region = image[mask].astype(np.uint8)
        if len(region) == 0:
            continue
        hist, _ = np.histogram(region, bins=256, range=(0, 255), density=True)
        hist += 1e-12
        total += -np.sum(hist * np.log(hist))
    return total

def add_texture_features(image):
    try:
        if image.dtype != np.uint8:
            image = img_as_ubyte((image - image.min()) / (image.max() - image.min()))

        glcm = graycomatrix(image,
                            distances=[1, 3, 5],
                            angles=[0, np.pi / 4, np.pi / 2],
                            levels=256,
                            symmetric=True,
                            normed=True)

        features = [
            image,
            np.full_like(image, graycoprops(glcm, 'contrast').mean()),
            np.full_like(image, graycoprops(glcm, 'energy').mean()),
            np.full_like(image, graycoprops(glcm, 'homogeneity').mean()),
            np.full_like(image, graycoprops(glcm, 'correlation').mean())
        ]

        return np.dstack(features)
    except Exception as e:
        print(f"Texture features skipped: {str(e)}")
        return np.expand_dims(image, axis=-1)

def pso_segmentation(image, n_thresholds):
    particles = np.random.randint(1, 255, (n_particles, n_thresholds))
    velocities = np.random.randn(n_particles, n_thresholds) * 10
    w_max, w_min = 0.9, 0.4

    pbest = particles.copy()
    pbest_scores = np.array([compute_fuzzy_entropy(image, p) for p in particles])
    gbest = pbest[np.argmax(pbest_scores)]
    gbest_score = np.max(pbest_scores)

    for iteration in range(n_iterations):
        w = w_max - (w_max - w_min) * (iteration / n_iterations) ** 0.7
        c1 = c1_initial - (2 * (iteration / n_iterations))
        c2 = c2_initial + (2 * (iteration / n_iterations))

        for i in range(n_particles):
            r1, r2 = np.random.rand(n_thresholds), np.random.rand(n_thresholds)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            velocities[i] = np.clip(velocities[i], -20, 20)
            particles[i] = np.clip(particles[i] + velocities[i], 1, 254).astype(int)
            particles[i] = np.sort(particles[i])

            current_score = compute_fuzzy_entropy(image, particles[i])
            if current_score > pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = current_score
                if current_score > gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = current_score

    return np.sort(gbest), gbest_score

def load_dataset(name='IndianPines'):
    os.makedirs('data', exist_ok=True)
    urls = {
        'IndianPines': (
            "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
            ('indian_pines_corrected', 'indian_pines_gt')
        ),
        'Salinas': (
            "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
            "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
            ('salinas_corrected', 'salinas_gt')
        ),
        'PaviaU': (
            "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
            ('paviaU', 'paviaU_gt')
        )
    }

    if name not in urls:
        raise ValueError(f"Unsupported dataset: {name}")

    url, gt_url, keys = urls[name]
    image_file = f'data/{name}_corrected.mat'
    gt_file = f'data/{name}_gt.mat'

    if not os.path.exists(image_file):
        print(f"Downloading {name} dataset...")
        urllib.request.urlretrieve(url, image_file)
    if not os.path.exists(gt_file):
        urllib.request.urlretrieve(gt_url, gt_file)

    return loadmat(image_file)[keys[0]], loadmat(gt_file)[keys[1]]

def create_train_mask(gt, train_ratio=0.1):
    train_mask = np.zeros_like(gt, dtype=bool)
    for cls in np.unique(gt):
        if cls == 0:
            continue
        indices = np.where(gt == cls)
        n_samples = max(1, int(len(indices[0]) * train_ratio))
        selected = np.random.choice(len(indices[0]), n_samples, replace=False)
        train_mask[indices[0][selected], indices[1][selected]] = True
    return train_mask

def evaluate_segmentation(original, segmented, gt, train_mask):
    data_range = original.max() - original.min()
    psnr_val = psnr(original, segmented, data_range=data_range)
    ssim_val = ssim(original, segmented, data_range=data_range)

    y = gt.ravel()
    X = segmented.ravel().reshape(-1, 1)
    train_mask_flat = train_mask.ravel()
    test_idx = ~train_mask_flat & (y != 0)

    if np.sum(test_idx) == 0:
        return 0, 0, 0, psnr_val, ssim_val

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X[train_mask_flat], y[train_mask_flat])
    y_pred = clf.predict(X[test_idx])

    oa = accuracy_score(y[test_idx], y_pred)
    kappa = cohen_kappa_score(y[test_idx], y_pred)

    cm = confusion_matrix(y[test_idx], y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_acc = np.nanmean(np.diag(cm) / np.sum(cm, axis=1))

    return oa, kappa, mean_acc, psnr_val, ssim_val

def main():
    datasets = ['IndianPines', 'Salinas', 'PaviaU']
    all_results = []

    for dataset_name in datasets:
        print(f"\n=== Processing {dataset_name} dataset ===")
        image, gt = load_dataset(dataset_name)

        h, w, d = image.shape

        # Step 1: Reduce spectral dimensions to 1 band (PC1)
        pca = PCA(n_components=1)
        reduced = pca.fit_transform(image.reshape(-1, d)).reshape(h, w)

        # Step 2: Normalize and convert to uint8
        reduced = ((reduced - reduced.min()) / (reduced.max() - reduced.min()) * 255).astype(np.uint8)

        # Step 3: Add texture features to reduced PC1 image
        features = add_texture_features(reduced)

        # Step 4: pc1 is used for applying thresholds (class assignment)
        pc1 = reduced

        # Step 5: Create training mask
        train_mask = create_train_mask(gt)
        dataset_results = []

        # Step 6: PSO segmentation and evaluation for k = 1 to 15
        for k in range(1, 16):
            print(f"\nRunning PSO for k={k} thresholds")
            thresholds, score = pso_segmentation(features, k)
            thresholds_list = [0] + [int(t) for t in thresholds] + [255]
            segmented = np.digitize(pc1, bins=thresholds_list[:-1])

            oa, kappa, mean_acc, psnr_val, ssim_val = evaluate_segmentation(
                pc1, segmented, gt, train_mask)

            print(f"k={k}: OA={oa:.4f}, Kappa={kappa:.4f}, PSNR={psnr_val:.2f}dB")
            dataset_results.append({
                'k': k, 'thresholds': thresholds_list, 'score': score,
                'OA': oa, 'Kappa': kappa, 'MeanAcc': mean_acc,
                'PSNR': psnr_val, 'SSIM': ssim_val
            })

        all_results.append({'dataset': dataset_name, 'results': dataset_results})

    # Step 7: Print final summary
    print("\n=== Final Results ===")
    for dataset in all_results:
        print(f"\nDataset: {dataset['dataset']}")
        print("k | OA | Kappa | MeanAcc | PSNR | SSIM")
        for result in dataset['results']:
            print(f"{result['k']:2d} | {result['OA']:.3f} | {result['Kappa']:.3f} | "
                  f"{result['MeanAcc']:.3f} | {result['PSNR']:.2f} | {result['SSIM']:.4f}")


if __name__ == "__main__":
    main()
