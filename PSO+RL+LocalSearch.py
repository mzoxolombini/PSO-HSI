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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from skimage.morphology import closing
from sklearn.svm import SVC
from scipy.ndimage import median_filter
from sklearn.model_selection import train_test_split


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

# PSO Parameters
n_particles = 30
n_iterations = 100
w_initial = 0.9
c1_initial = 2.5
c2_initial = 0.5

# RL Parameters
learning_rate = 0.1
discount_factor = 0.9

class RLAgent:
    def __init__(self):
        self.q_values = {
            'simple_hill': 1.0,
            'steepest_ascent': 1.0,
            'stochastic': 1.0,
            'first_choice': 1.0,
            'random_restart': 1.0
        }
        self.action_counts = {k: 0 for k in self.q_values.keys()}
        self.reward_history = []
        self.iteration = 0
        self.min_exploration = 0.1
        self.max_exploration = 0.5
        self.exploration_decay = 0.4

    def choose_action(self, n_iterations):
        exploration_rate = max(
            self.min_exploration,
            self.max_exploration - (self.iteration / n_iterations) * self.exploration_decay
        )
        if np.random.random() < exploration_rate:
            return np.random.choice(list(self.q_values.keys()))
        return max(self.q_values, key=self.q_values.get)

    def update_q_value(self, action, reward):
        self.q_values[action] += learning_rate * (
            reward + discount_factor * max(self.q_values.values()) - self.q_values[action]
        )
        self.action_counts[action] += 1
        self.reward_history.append(reward)
        self.iteration += 1

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

def simple_hill_climbing(image, current, current_score, max_neighbors=10):
    best = current.copy()
    best_score = current_score
    for _ in range(max_neighbors):
        neighbor = np.clip(current + np.random.randint(-5, 6, current.shape), 1, 254)
        neighbor = np.sort(neighbor)
        score = compute_fuzzy_entropy(image, neighbor)
        if score > best_score:
            best = neighbor
            best_score = score
            return best, best_score
    return best, best_score

def steepest_ascent_hill_climbing(image, current, current_score, k, max_neighbors=20):
    best = current.copy()
    best_score = current_score
    max_neighbors = 10 + k * 2
    for _ in range(max_neighbors):
        neighbor = np.clip(current + np.random.randint(-5, 6, current.shape), 1, 254)
        neighbor = np.sort(neighbor)
        score = compute_fuzzy_entropy(image, neighbor)
        if score > best_score:
            best = neighbor
            best_score = score
    return best, best_score

def stochastic_hill_climbing(image, current, current_score, iteration, max_iterations):
    neighbor = np.clip(current + np.random.randint(-5, 6, current.shape), 1, 254)
    neighbor = np.sort(neighbor)
    score = compute_fuzzy_entropy(image, neighbor)
    T = 1.0 - (iteration / max_iterations)
    if (score > current_score) or (np.random.rand() < np.exp(-(current_score - score) / (T + 1e-8))):
        return neighbor, score
    return current, current_score

def first_choice_hill_climbing(image, current, current_score, max_tries=20):
    for _ in range(max_tries):
        neighbor = np.clip(current + np.random.randint(-5, 6, current.shape), 1, 254)
        neighbor = np.sort(neighbor)
        score = compute_fuzzy_entropy(image, neighbor)
        if score > current_score:
            return neighbor, score
    return current, current_score

def random_restart_hill_climbing(image, current, current_score, restarts=3):
    best = current.copy()
    best_score = current_score
    for _ in range(restarts):
        neighbor = np.random.randint(1, 255, current.shape)
        neighbor = np.sort(neighbor)
        score = compute_fuzzy_entropy(image, neighbor)
        if score > best_score:
            best = neighbor
            best_score = score
    return best, best_score

def apply_rl_local_search(image, particles, scores, rl_agent, n_iterations, k):
    for i in range(len(particles)):
        action = rl_agent.choose_action(n_iterations)
        if action == 'simple_hill':
            new_p, new_score = simple_hill_climbing(image, particles[i], scores[i])
        elif action == 'steepest_ascent':
            new_p, new_score = steepest_ascent_hill_climbing(image, particles[i], scores[i], k)
        elif action == 'stochastic':
            new_p, new_score = stochastic_hill_climbing(image, particles[i], scores[i], rl_agent.iteration, n_iterations)
        elif action == 'first_choice':
            new_p, new_score = first_choice_hill_climbing(image, particles[i], scores[i])
        elif action == 'random_restart':
            new_p, new_score = random_restart_hill_climbing(image, particles[i], scores[i])

        reward = (new_score - scores[i]) / (scores[i] + 1e-8) * (1 + k / 15)
        rl_agent.update_q_value(action, reward)

        if new_score > scores[i]:
            particles[i] = new_p
            scores[i] = new_score

    return particles, scores



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

def pso_segmentation(image, n_thresholds, rl_agent):
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

        if iteration % 5 == 0:
            particles, pbest_scores = apply_rl_local_search(image, particles, pbest_scores, rl_agent, n_iterations, n_thresholds)

    return np.sort(gbest), gbest_score


def main():
    datasets = ['IndianPines', 'Salinas', 'PaviaU']
    all_results = []

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

        url, gt_url, keys = urls[name]
        image_file = f'data/{name}_corrected.mat'
        gt_file = f'data/{name}_gt.mat'

        if not os.path.exists(image_file):
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

    def evaluate_segmentation(segmented, pca_features, gt, train_mask):
        h, w, d = pca_features.shape
        segmented_flat = segmented.flatten().reshape(-1, 1)
        X = np.concatenate([pca_features.reshape(-1, d), segmented_flat], axis=1)
        y = gt.ravel()
        train_mask_flat = train_mask.ravel()
        test_idx = ~train_mask_flat & (y != 0)

        if np.sum(test_idx) == 0:
            return 0, 0, 0, 0, 0

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = SVC(kernel='rbf', C=10, gamma='scale')
        clf.fit(X_scaled[train_mask_flat], y[train_mask_flat])
        y_pred = clf.predict(X_scaled[test_idx])

        oa = accuracy_score(y[test_idx], y_pred)
        kappa = cohen_kappa_score(y[test_idx], y_pred)

        cm = confusion_matrix(y[test_idx], y_pred)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_acc = np.nanmean(np.diag(cm) / np.sum(cm, axis=1))

        return oa, kappa, mean_acc, 0, 0

    for dataset_name in datasets:
        print(f"\n=== Processing {dataset_name} dataset ===")
        image, gt = load_dataset(dataset_name)

        h, w, d = image.shape
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(image.reshape(-1, d)).reshape(h, w, 3)
        pc1 = features_pca[:, :, 0]  # Fix: needed for thresholding

        train_mask = create_train_mask(gt)
        dataset_results = []

        for k in range(1, 16):
            print(f"\nRunning PSO-RL for k={k} thresholds")
            rl_agent = RLAgent()
            thresholds, score = pso_segmentation(pc1, k, rl_agent)
            thresholds_list = [0] + [int(t) for t in thresholds] + [255]
            segmented = np.digitize(pc1, bins=thresholds_list[:-1])

            oa, kappa, mean_acc, psnr_val, ssim_val = evaluate_segmentation(
                segmented, features_pca, gt, train_mask)

            print(f"k={k}: OA={oa:.4f}, Kappa={kappa:.4f}")
            dataset_results.append({
                'k': k, 'thresholds': thresholds_list, 'score': score,
                'OA': oa, 'Kappa': kappa, 'MeanAcc': mean_acc,
                'PSNR': psnr_val, 'SSIM': ssim_val
            })

        all_results.append({'dataset': dataset_name, 'results': dataset_results})

    print("\n=== Final Results ===")
    for dataset in all_results:
        print(f"\nDataset: {dataset['dataset']}")
        print("k | OA | Kappa | MeanAcc | PSNR | SSIM")
        for result in dataset['results']:
            print(f"{result['k']:2d} | {result['OA']:.3f} | {result['Kappa']:.3f} | "
                  f"{result['MeanAcc']:.3f} | {result['PSNR']:.2f} | {result['SSIM']:.4f}")

if __name__ == "__main__":
    main()
