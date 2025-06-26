import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from sklearn.metrics import accuracy_score, cohen_kappa_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import warnings
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

warnings.filterwarnings("ignore")

# PSO Parameters
n_particles = 30
n_iterations = 50
w, c1, c2 = 0.7, 1.5, 1.5

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


def stochastic_hill_climbing(image, current, current_score):
    neighbor = np.clip(current + np.random.randint(-5, 6, current.shape), 1, 254)
    neighbor = np.sort(neighbor)
    score = compute_fuzzy_entropy(image, neighbor)
    if score > current_score:
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
    improved = False
    for i in range(len(particles)):
        action = rl_agent.choose_action(n_iterations)

        if action == 'simple_hill':
            new_p, new_score = simple_hill_climbing(image, particles[i], scores[i])
        elif action == 'steepest_ascent':
            new_p, new_score = steepest_ascent_hill_climbing(image, particles[i], scores[i], k)
        elif action == 'stochastic':
            new_p, new_score = stochastic_hill_climbing(image, particles[i], scores[i])
        elif action == 'first_choice':
            new_p, new_score = first_choice_hill_climbing(image, particles[i], scores[i])
        elif action == 'random_restart':
            new_p, new_score = random_restart_hill_climbing(image, particles[i], scores[i])

        reward = (new_score - scores[i]) / (scores[i] + 1e-8) * (1 + k / 15)
        rl_agent.update_q_value(action, reward)

        if new_score > scores[i]:
            particles[i] = new_p
            scores[i] = new_score
            improved = True

    return particles, scores, improved


def pso_segmentation(image, n_thresholds, rl_agent):
    particles = np.random.randint(1, 255, (n_particles, n_thresholds))
    velocities = np.random.randn(n_particles, n_thresholds) * 10

    pbest = particles.copy()
    pbest_scores = np.array([compute_fuzzy_entropy(image, p) for p in particles])
    gbest = pbest[np.argmax(pbest_scores)]
    gbest_score = np.max(pbest_scores)

    for iteration in range(n_iterations):
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
            particles, pbest_scores, improved = apply_rl_local_search(
                image, particles, pbest_scores, rl_agent, n_iterations, n_thresholds)

            if len(rl_agent.reward_history) >= 10:
                recent_rewards = rl_agent.reward_history[-10:]
                if (np.mean(recent_rewards) < 0.001 and
                        np.std(recent_rewards) < 0.005):
                    print(f"Converged at iteration {iteration}")
                    break
            elif iteration > n_iterations // 2 and gbest_score - pbest_scores.mean() < 0.01:
                print(f"Stagnation detected at iteration {iteration}")
                break

    return np.sort(gbest), gbest_score


def load_dataset(name='IndianPines'):
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
        print(f"Downloading {name} dataset...")
        urllib.request.urlretrieve(url, image_file)
    if not os.path.exists(gt_file):
        urllib.request.urlretrieve(gt_url, gt_file)

    image = loadmat(image_file)[img_key]
    gt = loadmat(gt_file)[gt_key]
    return image, gt


def create_train_mask(gt, train_ratio=0.1):
    train_mask = np.zeros_like(gt, dtype=bool)
    classes = np.unique(gt)
    classes = classes[classes != 0]  # Exclude background

    for cls in classes:
        indices = np.where(gt == cls)
        n_samples = int(len(indices[0]) * train_ratio)
        selected = np.random.choice(len(indices[0]), n_samples, replace=False)
        train_mask[indices[0][selected], indices[1][selected]] = True

    return train_mask


def evaluate_segmentation(original, segmented, gt, train_mask):
    # Calculate PSNR and SSIM
    data_range = original.max() - original.min()
    psnr_val = psnr(original, segmented, data_range=data_range)
    ssim_val = ssim(original, segmented, data_range=data_range)

    # Prepare classification data
    le = LabelEncoder()
    y = gt.ravel()
    X = segmented.ravel().reshape(-1, 1)

    # Split into train/test
    train_idx = train_mask.ravel()
    test_idx = ~train_mask & (y != 0)  # Exclude background and training pixels

    if np.sum(test_idx) == 0:
        return 0, 0, 0, psnr_val, ssim_val

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Simple classifier (you can replace with your preferred classifier)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculate metrics
    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    # Calculate mean accuracy
    cm = confusion_matrix(y_test, y_pred)
    mean_acc = np.mean(np.diag(cm) / np.sum(cm, axis=1))

    return oa, kappa, mean_acc, psnr_val, ssim_val


def main():
    datasets = ['IndianPines', 'Salinas', 'PaviaU']
    all_results = []

    for dataset_name in datasets:
        print(f"\n=== Processing {dataset_name} dataset ===")
        image, gt = load_dataset(dataset_name)

        h, w, d = image.shape
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(image.reshape(-1, d))
        pc1 = reduced[:, 0].reshape(h, w)
        pc1 = ((pc1 - pc1.min()) / (pc1.max() - pc1.min()) * 255).astype(np.uint8)

        # Create train mask
        train_mask = create_train_mask(gt)

        dataset_results = []
        for k in range(1, 16):  # k from 1 to 15
            rl_agent = RLAgent()  # New agent for each k
            print(f"\nRunning PSO-RL for k={k} thresholds")
            thresholds, score = pso_segmentation(pc1, k, rl_agent)
            thresholds_list = [0] + [int(t) for t in thresholds] + [255]

            # Apply thresholds to create segmented image
            segmented = np.digitize(pc1, bins=thresholds_list[:-1])

            # Evaluate segmentation
            oa, kappa, mean_acc, psnr_val, ssim_val = evaluate_segmentation(
                pc1, segmented, gt, train_mask)

            print(f"Optimal thresholds: {thresholds_list}")
            print(f"Fitness score: {score:.4f}")
            print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}, Mean Acc: {mean_acc:.4f}")
            print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

            dataset_results.append({
                'k': k,
                'thresholds': thresholds_list,
                'score': score,
                'OA': oa,
                'Kappa': kappa,
                'MeanAcc': mean_acc,
                'PSNR': psnr_val,
                'SSIM': ssim_val
            })

        all_results.append({
            'dataset': dataset_name,
            'results': dataset_results
        })

    # Print all results
    print("\n=== Final Results ===")
    for dataset in all_results:
        print(f"\nDataset: {dataset['dataset']}")
        print("k | Thresholds | Score | OA | Kappa | MeanAcc | PSNR | SSIM")
        print("-" * 80)
        for result in dataset['results']:
            print(f"{result['k']:2d} | {result['thresholds']} | {result['score']:.4f} | "
                  f"{result['OA']:.4f} | {result['Kappa']:.4f} | {result['MeanAcc']:.4f} | "
                  f"{result['PSNR']:.2f} | {result['SSIM']:.4f}")


if __name__ == "__main__":
    main()