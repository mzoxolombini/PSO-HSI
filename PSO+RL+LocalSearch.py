import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from skimage.util import img_as_ubyte
from skimage.filters.rank import gradient
from skimage.morphology import disk
import warnings
import os
import urllib.request

warnings.filterwarnings("ignore")

# PSO Parameters
n_particles = 30
n_iterations = 50
w, c1, c2 = 0.7, 1.5, 1.5

# RL Parameters
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = max(0.1, 0.5 - (iter/n_iterations)*0.4)


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
        self.iteration = 0  # Add iteration counter

    def choose_action(self, n_iterations):
        # Adaptive exploration rate
        exploration_rate = max(0.1, 0.5 - (self.iteration / n_iterations) * 0.4)
        if np.random.random() < exploration_rate:
            return np.random.choice(list(self.q_values.keys()))
        return max(self.q_values, key=self.q_values.get)

    def update_q_value(self, action, reward):
        self.q_values[action] += learning_rate * (
                reward + discount_factor * max(self.q_values.values()) - self.q_values[action]
        )
        self.action_counts[action] += 1
        self.reward_history.append(reward)
        self.iteration += 1  # Increment iteration counter


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
            return best, best_score  # Return first improvement
    return best, best_score


def steepest_ascent_hill_climbing(image, current, current_score, max_neighbors=10 + k*2):
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


def apply_rl_local_search(image, particles, scores, rl_agent):
    improved = False
    for i in range(len(particles)):
        action = rl_agent.choose_action()

        if action == 'simple_hill':
            new_p, new_score = simple_hill_climbing(image, particles[i], scores[i])
        elif action == 'steepest_ascent':
            new_p, new_score = steepest_ascent_hill_climbing(image, particles[i], scores[i])
        elif action == 'stochastic':
            new_p, new_score = stochastic_hill_climbing(image, particles[i], scores[i])
        elif action == 'first_choice':
            new_p, new_score = first_choice_hill_climbing(image, particles[i], scores[i])
        elif action == 'random_restart':
            new_p, new_score = random_restart_hill_climbing(image, particles[i], scores[i])

        reward = (new_score - scores[i]) / (scores[i] + 1e-8) * (1 + k/15)
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
        # Standard PSO updates
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

        # RL-based local search every 5 iterations
        if iter % 5 == 0:
            particles, pbest_scores, improved = apply_rl_local_search(
                image, particles, pbest_scores, rl_agent)

            # Enhanced early stopping conditions
            if len(rl_agent.reward_history) >= 10:
                recent_rewards = rl_agent.reward_history[-10:]
                if (np.mean(recent_rewards) < 0.001 and
                        np.std(recent_rewards) < 0.005):
                    print(f"Converged at iteration {iter}")
                    break
            elif iter > n_iterations // 2 and gbest_score - pbest_scores.mean() < 0.01:
                print(f"Stagnation detected at iteration {iter}")
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


def main():
    rl_agent = RLAgent()
    datasets = ['IndianPines', 'Salinas', 'PaviaU']

    for dataset_name in datasets:
        print(f"\n=== Processing {dataset_name} dataset ===")
        image, gt = load_dataset(dataset_name)

        # Dimensionality reduction
        h, w, d = image.shape
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(image.reshape(-1, d))
        pc1 = reduced[:, 0].reshape(h, w)
        pc1 = ((pc1 - pc1.min()) / (pc1.max() - pc1.min()) * 255).astype(np.uint8)

        for k in [5, 10, 15]:
            print(f"\nRunning PSO-RL for k={k} thresholds")
            thresholds, score = pso_segmentation(pc1, k, rl_agent)
            print(f"Optimal thresholds: {[0] + [int(t) for t in thresholds] + [255]}")
            print(f"Fitness score: {score:.4f}")

            # Print RL statistics
            print("\nRL Agent Performance:")
            total_actions = sum(rl_agent.action_counts.values())
            for action, count in rl_agent.action_counts.items():
                print(f"{action:15s}: {count:3d} uses ({count / total_actions:.1%})")
            print(f"Average reward: {np.mean(rl_agent.reward_history):.4f}")


if __name__ == "__main__":
    main()