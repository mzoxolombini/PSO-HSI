import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from pathlib import Path
import warnings
import pandas as pd
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# --- EMAP + SVM Baseline ---
def emap_svm_baseline(img, gt):
    reshaped = img.reshape(-1, img.shape[-1])
    pca = PCA(n_components=10)
    reduced = pca.fit_transform(reshaped)
    scaler = StandardScaler()
    reduced_scaled = scaler.fit_transform(reduced)

    flat_labels = gt.flatten()
    mask = flat_labels > 0

    clf = SVC(kernel='rbf')
    clf.fit(reduced_scaled[mask], flat_labels[mask])
    predicted = clf.predict(reduced_scaled)
    predicted = predicted.reshape(gt.shape)
    oa = accuracy_score(gt.flatten(), predicted.flatten())
    kappa = cohen_kappa_score(gt.flatten(), predicted.flatten())
    return oa, kappa

# --- LBP + SVM Baseline ---
def lbp_svm_baseline(gray_img, gt):
    lbp = local_binary_pattern(img_as_ubyte(gray_img), P=8, R=1, method='uniform')
    lbp_flat = lbp.reshape(-1, 1)

    flat_labels = gt.flatten()
    mask = flat_labels > 0

    clf = SVC(kernel='rbf')
    clf.fit(lbp_flat[mask], flat_labels[mask])
    predicted = clf.predict(lbp_flat)
    predicted = predicted.reshape(gt.shape)
    oa = accuracy_score(gt.flatten(), predicted.flatten())
    kappa = cohen_kappa_score(gt.flatten(), predicted.flatten())
    return oa, kappa

# --- Composite Kernel SVM (SVM-CK) ---
def svm_ck_baseline(img, gt):
    reshaped = img.reshape(-1, img.shape[-1])
    pca = PCA(n_components=10)
    reduced = pca.fit_transform(reshaped)
    spatial_mean = np.mean(reduced, axis=1).reshape(-1, 1)
    features = np.hstack((reduced, spatial_mean))
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    flat_labels = gt.flatten()
    mask = flat_labels > 0

    clf = SVC(kernel='rbf')
    clf.fit(features_scaled[mask], flat_labels[mask])
    predicted = clf.predict(features_scaled)
    predicted = predicted.reshape(gt.shape)
    oa = accuracy_score(gt.flatten(), predicted.flatten())
    kappa = cohen_kappa_score(gt.flatten(), predicted.flatten())
    return oa, kappa

# --- Main Evaluation Entry Point ---
def evaluate_baselines(img, gray_img, gt):
    print("\n-- Evaluating Non-Neural SOTA Baselines --")
    emap_oa, emap_kappa = emap_svm_baseline(img, gt)
    print(f"EMAP+SVM -> OA: {emap_oa:.4f}, Kappa: {emap_kappa:.4f}")

    lbp_oa, lbp_kappa = lbp_svm_baseline(gray_img, gt)
    print(f"LBP+SVM -> OA: {lbp_oa:.4f}, Kappa: {lbp_kappa:.4f}")

    svm_ck_oa, svm_ck_kappa = svm_ck_baseline(img, gt)
    print(f"SVM-CK -> OA: {svm_ck_oa:.4f}, Kappa: {svm_ck_kappa:.4f}")

    return {
        'EMAP+SVM': (emap_oa, emap_kappa),
        'LBP+SVM': (lbp_oa, lbp_kappa),
        'SVM-CK': (svm_ck_oa, svm_ck_kappa)
    }

# --- Main Function ---
def main():
    data_dir = Path(r"C:\Users\mzoxo\OneDrive\Documents\data")
    mat_files = list(data_dir.glob("*.mat"))

    gt_files = {f.stem.lower(): f for f in mat_files if "_gt" in f.stem.lower()}
    img_files = [f for f in mat_files if "_gt" not in f.stem.lower()]

    for img_path in img_files:
        dataset_name = img_path.stem.lower().replace("_corrected", "")
        dataset_title = dataset_name.replace("_", " ").title()
        print(f"\n=== Processing dataset: {dataset_title} ===")

        img_data = loadmat(img_path)
        img_key = [k for k in img_data.keys() if not k.startswith('__')][0]
        img = img_data[img_key]

        reshaped = img.reshape(-1, img.shape[-1])
        pca = PCA(n_components=3)
        pc_img = pca.fit_transform(reshaped).reshape(*img.shape[:2], 3)
        gray_pca = np.mean(pc_img, axis=-1)

        gray_norm = (gray_pca - gray_pca.min()) / (gray_pca.max() - gray_pca.min())

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
            evaluate_baselines(img, gray_norm, gt)

if __name__ == "__main__":
    main()
