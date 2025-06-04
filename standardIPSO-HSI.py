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
from skimage.util import img_as_ubyte
from skimage.filters.rank import gradient
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for smoother execution


def compute_between_class_variance(thresholds, hist):
    total_variance = 0
    for i in range(len(thresholds) - 1):
        t1, t2 = int(thresholds[i]), int(thresholds[i + 1])
        if t1 >= t2:
            continue
        w0 = hist[t1:t2].sum()
        if w0 == 0:
            continue
        mu0 = np.sum(np.arange(t1, t2) * hist[t1:t2]) / w0
        total_variance += w0 * mu0 ** 2
    return total_variance


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
    segmented = np.zeros_like(gray, dtype=np.uint8)
    for i in range(len(thresholds) - 1):
        mask = (gray >= thresholds[i]) & (gray < thresholds[i + 1])
        segmented[mask] = i + 1
    segmented = median_filter(segmented, size=3)
    segmented = opening(segmented, disk(1))
    segmented = closing(segmented, disk(2))
    return segmented


def calculate_metrics(original, segmented, ground_truth, train_mask):
    psnr_val = psnr(original, segmented, data_range=255)
    ssim_val = ssim(original, segmented, data_range=255)

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
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SVC(kernel='rbf', C=10, gamma='auto', class_weight='balanced')
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


def main():
    image, gt, train_mask = load_real_data()
    pca = PCA(n_components=1)
    gray = pca.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape[0], image.shape[1])
    gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)

    for k in [10, 12, 14]:
        print(f"\nRunning original IPSO for {k} levels...")

        thresholds = np.linspace(0, 255, k + 1).astype(int)
        segmented = segment_image(gray, thresholds)
        psnr_val, ssim_val, oa, kappa = calculate_metrics(gray, segmented, gt, train_mask)

        print(f"\nResults for {k} levels:")
        print("Thresholds:", thresholds.tolist())
        print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
        print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}")

        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.imshow(gray, cmap='gray')
        plt.title('PCA Component')
        plt.subplot(132)
        plt.imshow(segmented)
        plt.title('Segmentation')
        plt.subplot(133)
        plt.imshow(gt)
        plt.title('Ground Truth')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
