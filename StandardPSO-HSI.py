import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
import kagglehub
import os
from sklearn.decomposition import PCA
import cv2
import pandas as pd
import ace_tools_open as tools

# Load dataset
path = kagglehub.dataset_download("abhijeetgo/indian-pines-hyperspectral-dataset")
image = np.load(os.path.join(path, 'indianpinearray.npy'))
ground_truth = np.load(os.path.join(path, 'IPgt.npy'))

# Apply PCA to reduce to 1 band
pca = PCA(n_components=1)
reshaped = image.reshape(-1, image.shape[2])
image_pca = pca.fit_transform(reshaped).reshape(image.shape[0], image.shape[1])
image_pca = cv2.normalize(image_pca, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Fuzzy Entropy Objective Function
def fuzzy_entropy(thresholds, image):
    thresholds = np.sort(np.round(thresholds).astype(np.uint8))
    levels = np.concatenate(([0], thresholds, [255]))
    fuzzy_ent = 0
    for i in range(len(levels)-1):
        region = image[(image >= levels[i]) & (image < levels[i+1])]
        if region.size == 0:
            continue
        P = region / 255.0
        entropy = -np.sum(P * np.log(P + 1e-10) + (1 - P) * np.log(1 - P + 1e-10))
        fuzzy_ent += entropy
    return -fuzzy_ent

# Particle Swarm Optimization
def pso(objective, image, num_thresholds=5, num_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5):
    dim = num_thresholds
    lb, ub = 0, 255
    X = np.random.uniform(lb, ub, (num_particles, dim))
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_scores = np.array([objective(p, image) for p in pbest])
    gbest = pbest[np.argmin(pbest_scores)]
    gbest_score = np.min(pbest_scores)
    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] = np.clip(X[i] + V[i], lb, ub)
            score = objective(X[i], image)
            if score < pbest_scores[i]:
                pbest[i] = X[i]
                pbest_scores[i] = score
        if np.min(pbest_scores) < gbest_score:
            gbest = pbest[np.argmin(pbest_scores)]
            gbest_score = np.min(pbest_scores)
    return np.sort(np.round(gbest).astype(np.uint8)), -gbest_score

# Run PSO
thresholds, best_entropy = pso(fuzzy_entropy, image_pca, num_thresholds=5)
levels = np.concatenate(([0], thresholds, [255]))
segmented = np.zeros_like(image_pca)
for i in range(len(levels)-1):
    segmented[(image_pca >= levels[i]) & (image_pca < levels[i+1])] = i + 1

# Evaluation Metrics
flat_pred = segmented.flatten()
flat_gt = ground_truth.flatten()
mask = flat_gt > 0
flat_pred = flat_pred[mask]
flat_gt = flat_gt[mask]

oa = accuracy_score(flat_gt, flat_pred)
kappa = cohen_kappa_score(flat_gt, flat_pred)
class_accuracies = [accuracy_score(flat_gt[flat_gt == label], flat_pred[flat_gt == label]) for label in np.unique(flat_gt)]
aa = np.mean(class_accuracies)

# Display metrics
tools.display_dataframe_to_user(name="Segmentation Evaluation Metrics", dataframe=pd.DataFrame({
    "Metric": ["Overall Accuracy", "Average Accuracy", "Kappa Coefficient"],
    "Value": [oa, aa, kappa]
}))
