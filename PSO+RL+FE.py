import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from skimage.filters import gabor
from skimage.morphology import disk, binary_opening, binary_closing, erosion, dilation
from skimage.util import img_as_ubyte
from pathlib import Path
import warnings
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import slic
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import generic_filter
from skimage.exposure import rescale_intensity

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

warnings.filterwarnings("ignore")

n_particles = 30
n_iterations = 50
w, c1, c2 = 0.7, 1.5, 1.5
F, CR, n_elites = 0.8, 0.95, 10

RL_strategy_log = []
DE_improvements = []

def extract_features(image):
    features, base_shape = [], image.shape
    for theta in range(4):
        for sigma in (1, 3):
            for freq in (0.05, 0.25):
                real, imag = gabor(image, frequency=freq, theta=theta / 4. * np.pi, sigma_x=sigma, sigma_y=sigma)
                if real.shape == base_shape: features.append(real)
                if imag.shape == base_shape: features.append(imag)
    try:
        image_rescaled = rescale_intensity(image, in_range='image', out_range=(0, 1))
        img_u8 = img_as_ubyte(np.clip(image_rescaled, 0, 1))
        glcm = graycomatrix(img_u8, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            features.append(np.full(base_shape, graycoprops(glcm, prop)[0, 0]))
    except Exception as e:
        print(f"GLCM skipped: {e}")
    for r in (3, 5):
        selem = disk(r)
        features.extend([erosion(image, selem), dilation(image, selem)])
    return np.stack(features, axis=-1)

def fuzzy_entropy(image, thresholds):
    thresholds = np.sort(thresholds)
    thresholds = np.concatenate(([0], thresholds, [256]))
    total = 0
    for i in range(len(thresholds) - 1):
        mask = (image >= thresholds[i]) & (image < thresholds[i+1])
        region = image[mask].astype(np.uint8)
        if len(region) == 0: continue
        hist, _ = np.histogram(region, bins=256, range=(0, 255), density=True)
        hist += 1e-12
        total += -np.sum(hist * np.log(hist))
    return total

def choose_action():
    return np.random.choice(['greedy', 'tabu', 'annealing', 'random'], p=[0.4, 0.2, 0.2, 0.2])

def rl_local_search(image, population, scores):
    action = choose_action()
    RL_strategy_log.append(action)
    if action == 'greedy':
        for i in range(len(population)):
            for d in range(population.shape[1]):
                candidate = population[i].copy()
                candidate[d] = np.clip(candidate[d] + np.random.choice([-1, 1]), 1, 254)
                score = fuzzy_entropy(image, candidate)
                if score > scores[i]: population[i], scores[i] = candidate, score
    elif action == 'tabu':
        visited = set()
        for i in range(len(population)):
            candidate = np.clip(population[i] + np.random.randint(-3, 4, population.shape[1]), 1, 254)
            key = tuple(candidate)
            if key not in visited:
                visited.add(key)
                score = fuzzy_entropy(image, candidate)
                if score > scores[i]: population[i], scores[i] = candidate, score
    elif action == 'annealing':
        T, T_min, alpha = 1.0, 1e-3, 0.9
        for i in range(len(population)):
            candidate = np.clip(population[i] + np.random.randint(-5, 6, population.shape[1]), 1, 254)
            score = fuzzy_entropy(image, candidate)
            if score > scores[i] or np.random.rand() < np.exp((score - scores[i]) / (T + 1e-8)):
                population[i], scores[i] = candidate, score
            T = max(T_min, alpha * T)
    elif action == 'random':
        population = np.random.randint(1, 255, population.shape)
        scores = np.array([fuzzy_entropy(image, p) for p in population])
    return population, scores

def hybrid_de(image, population, scores):
    elite_idx = np.argsort(scores)[-n_elites:]
    elites = population[elite_idx].copy()
    improvements = 0
    for i in range(n_elites):
        idxs = [j for j in range(n_elites) if j != i]
        a, b, c = elites[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + F * (b - c), 1, 254)
        cross = np.random.rand(mutant.shape[0]) < CR
        if not np.any(cross): cross[np.random.randint(0, mutant.shape[0])] = True
        trial = np.where(cross, mutant, elites[i]).astype(int)
        score = fuzzy_entropy(image, trial)
        if score > scores[elite_idx[i]]:
            population[elite_idx[i]], scores[elite_idx[i]] = trial, score
            improvements += 1
    DE_improvements.append(improvements)
    return population, scores

def pso_segmentation(image, n_thresholds):
    segments = slic(image, n_segments=n_thresholds * 5, compactness=10, channel_axis=None)
    super_vals = [np.mean(image[segments == i]) for i in np.unique(segments)]
    init_thresh = np.percentile(super_vals, np.linspace(0, 100, n_thresholds + 2)[1:-1])

    image = image.flatten()
    dim, lb, ub = n_thresholds, 0, 255
    particles = np.random.randint(lb, ub, (n_particles, dim))
    particles[0] = init_thresh
    velocities = np.random.randn(n_particles, dim)

    pbest, pbest_scores = particles.copy(), np.array([fuzzy_entropy(image, p) for p in particles])
    gbest = pbest[np.argmax(pbest_scores)]

    for iter in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i]))
            particles[i] = np.clip(particles[i] + velocities[i], lb, ub).astype(int)
            score = fuzzy_entropy(image, particles[i])
            if score > pbest_scores[i]:
                pbest[i], pbest_scores[i] = particles[i], score
                if score > fuzzy_entropy(image, gbest): gbest = particles[i]
        if iter % 5 == 0:
            particles, pbest_scores = rl_local_search(image, particles, pbest_scores)
            particles, pbest_scores = hybrid_de(image, particles, pbest_scores)
            gbest = particles[np.argmax(pbest_scores)]
    return np.sort(gbest)

def apply_thresholds(image, thresholds):
    thresholds = np.sort(thresholds)
    result = np.zeros_like(image)
    for i, t in enumerate(thresholds): result[image > t] = i + 1
    return result

svm = SVC(kernel='rbf', C=100, gamma=0.01, decision_function_shape='ovr')

def main():
    dataset_path = "C:/Users/mzoxo/OneDrive/Documents/data/Indian_pines_corrected.mat"
    gt_path = "C:/Users/mzoxo/OneDrive/Documents/data/Indian_pines_gt.mat"
    image = loadmat(dataset_path)['indian_pines_corrected'].astype(np.float32)
    gt = loadmat(gt_path)['indian_pines_gt']

    h, w, d = image.shape
    image_2d = image.reshape(-1, d)

    pca = PCA(n_components=30)
    pca_result = pca.fit_transform(image_2d)
    reduced = pca_result[:, 0].reshape(h, w)

    results = []
    for k in range(1, 15):
        print(f"-- Running for k = {k} thresholds")
        thresholds = pso_segmentation(reduced, k)
        segmented = apply_thresholds(reduced, thresholds)
        features = extract_features(segmented)
        features = features.reshape(-1, features.shape[-1])
        labels = gt.reshape(-1)

        idx = labels > 0
        X, y = features[idx], labels[idx]
        X = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, stratify=y, random_state=42)

        from collections import Counter
        min_class_size = min(Counter(y_train).values())
        k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
        smote = SMOTE(k_neighbors=k_neighbors)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        svm.fit(X_train_res, y_train_res)
        y_pred = svm.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        print(f"OA: {acc:.4f} | Kappa: {kappa:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        results.append({"k": k, "OA": acc, "Kappa": kappa})

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("PSO_RL_FE_SVM_results.csv", index=False)

    plt.plot(df["k"], df["OA"], marker='o', label="OA")
    plt.plot(df["k"], df["Kappa"], marker='s', label="Kappa")
    plt.title("PSO+RL+FE+SVM-CK Performance (Indian Pines)")
    plt.xlabel("Thresholds (k)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
