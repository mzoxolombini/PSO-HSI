import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time


class IPSO:
    def __init__(self, image, k_levels=10, population_size=50, max_iter=100,
                 w_initial=1.2, c_x=1.5, c_y=1.5, replace_ratio=0.5):
        """
        Initialize IPSO parameters

        Args:
            image: Input hyperspectral image (3D numpy array)
            k_levels: Number of segmentation levels
            population_size: Number of particles in each swarm
            max_iter: Maximum number of iterations
            w_initial: Initial inertia weight
            c_x, c_y: Acceleration coefficients
            replace_ratio: Ratio of worst particles to replace
        """
        self.image = image
        self.k_levels = k_levels
        self.pop_size = population_size
        self.max_iter = max_iter
        self.w_initial = w_initial
        self.c_x = c_x
        self.c_y = c_y
        self.replace_ratio = replace_ratio

        # Convert RGB to L*a*b* color space
        self.lab_image = color.rgb2lab(image)

        # Number of fuzzy parameters (2*(k-1))
        self.D = 2 * (k_levels - 1)

        # Initialize swarms
        self.swarms = []
        for _ in range(self.D):
            swarm = {
                'positions': np.random.uniform(0, 255, (population_size, 1)),
                'velocities': np.zeros((population_size, 1)),
                'pb_positions': np.zeros((population_size, 1)),
                'pb_fitness': np.full(population_size, -np.inf),
                'gb_position': np.zeros(1),
                'gb_fitness': -np.inf,
                'stagnation_count': 0
            }
            self.swarms.append(swarm)

        # Context parameter (CP) - will store gb positions from other dimensions
        self.CP = np.zeros(self.D)

        # Store best fitness history
        self.best_fitness_history = []

    def trapezoidal_membership(self, n, a, b):
        """Trapezoidal membership function"""
        if n <= a:
            return 1.0
        elif a < n <= b:
            return (b - n) / (b - a)
        else:
            return 0.0

    def calculate_fuzzy_entropy(self, thresholds):
        """Calculate fuzzy entropy for given thresholds"""
        # Convert thresholds to fuzzy parameters (a1, b1, a2, b2, ...)
        fuzzy_params = []
        for i in range(self.k_levels - 1):
            a = thresholds[2 * i]
            b = thresholds[2 * i + 1]
            fuzzy_params.extend([a, b])

        # Calculate histogram (combining all channels)
        hist, _ = np.histogram(self.lab_image.ravel(), bins=256, range=(0, 255))
        hist = hist.astype(float) / hist.sum()  # Normalize

        total_entropy = 0.0

        for k in range(1, self.k_levels + 1):
            if k == 1:
                a_prev, b_prev = 0, 0
                a, b = fuzzy_params[0], fuzzy_params[1]
                R_k = 0
                for i in range(256):
                    mu = self.trapezoidal_membership(i, a, b)
                    R_k += hist[i] * mu

                if R_k == 0:
                    H_k = 0
                else:
                    H_k = 0
                    for i in range(256):
                        mu = self.trapezoidal_membership(i, a, b)
                        if mu > 0:
                            term = (hist[i] * mu) / R_k
                            if term > 0:
                                H_k -= term * np.log(term)

            elif k == self.k_levels:
                a_prev, b_prev = fuzzy_params[-2], fuzzy_params[-1]
                R_k = 0
                for i in range(256):
                    mu = self.trapezoidal_membership(i, a_prev, b_prev)
                    R_k += hist[i] * mu

                if R_k == 0:
                    H_k = 0
                else:
                    H_k = 0
                    for i in range(256):
                        mu = self.trapezoidal_membership(i, a_prev, b_prev)
                        if mu > 0:
                            term = (hist[i] * mu) / R_k
                            if term > 0:
                                H_k -= term * np.log(term)

            else:
                a_prev, b_prev = fuzzy_params[2 * (k - 2)], fuzzy_params[2 * (k - 2) + 1]
                a, b = fuzzy_params[2 * (k - 1)], fuzzy_params[2 * (k - 1) + 1]

                R_k = 0
                for i in range(256):
                    if i <= a_prev:
                        mu = 0
                    elif a_prev < i <= b_prev:
                        mu = (i - a_prev) / (b_prev - a_prev)
                    elif b_prev < i <= a:
                        mu = 1
                    elif a < i <= b:
                        mu = (b - i) / (b - a)
                    else:
                        mu = 0
                    R_k += hist[i] * mu

                if R_k == 0:
                    H_k = 0
                else:
                    H_k = 0
                    for i in range(256):
                        if i <= a_prev:
                            mu = 0
                        elif a_prev < i <= b_prev:
                            mu = (i - a_prev) / (b_prev - a_prev)
                        elif b_prev < i <= a:
                            mu = 1
                        elif a < i <= b:
                            mu = (b - i) / (b - a)
                        else:
                            mu = 0

                        if mu > 0:
                            term = (hist[i] * mu) / R_k
                            if term > 0:
                                H_k -= term * np.log(term)

            total_entropy += H_k

        return total_entropy

    def evaluate_fitness(self, particle_positions, dim):
        """Evaluate fitness for particles in a given dimension"""
        # Create context parameter for this evaluation
        temp_CP = self.CP.copy()
        temp_CP[dim] = particle_positions

        # Calculate fitness (fuzzy entropy)
        fitness = self.calculate_fuzzy_entropy(temp_CP)

        return fitness

    def update_velocity_position(self, swarm_idx):
        """Update velocity and position for particles in a swarm"""
        swarm = self.swarms[swarm_idx]

        for i in range(self.pop_size):
            # Tournament selection to choose which pb to follow
            if np.random.rand() > 0.5:  # selection probability = 0.5
                # Follow own pb
                f = i
            else:
                # Tournament selection: pick 2 random particles, choose better one
                candidates = np.random.choice(self.pop_size, 2, replace=False)
                if swarm['pb_fitness'][candidates[0]] > swarm['pb_fitness'][candidates[1]]:
                    f = candidates[0]
                else:
                    f = candidates[1]

            # Update velocity
            w = self.w_initial - (self.w_initial / self.max_iter) * self.iteration
            r1, r2 = np.random.rand(), np.random.rand()

            cognitive = self.c_x * r1 * (swarm['pb_positions'][f] - swarm['positions'][i])
            social = self.c_y * r2 * (swarm['gb_position'] - swarm['positions'][i])

            swarm['velocities'][i] = w * swarm['velocities'][i] + cognitive + social

            # Limit velocity
            v_max = 10.0  # arbitrary limit
            swarm['velocities'][i] = np.sign(swarm['velocities'][i]) * \
                                     min(abs(swarm['velocities'][i]), v_max)

            # Update position
            swarm['positions'][i] += swarm['velocities'][i]

            # Ensure positions stay within bounds [0, 255]
            swarm['positions'][i] = np.clip(swarm['positions'][i], 0, 255)

    def replace_worst_particles(self, swarm_idx):
        """Replace worst particles with clones of best particles"""
        swarm = self.swarms[swarm_idx]
        num_replace = int(self.pop_size * self.replace_ratio)

        # Get indices of best and worst particles
        fitness = swarm['pb_fitness']
        best_indices = np.argsort(fitness)[-num_replace:]
        worst_indices = np.argsort(fitness)[:num_replace]

        # Replace worst particles with clones of best particles
        swarm['positions'][worst_indices] = swarm['positions'][best_indices].copy()
        swarm['velocities'][worst_indices] = np.zeros_like(swarm['velocities'][best_indices])
        swarm['pb_positions'][worst_indices] = swarm['pb_positions'][best_indices].copy()
        swarm['pb_fitness'][worst_indices] = swarm['pb_fitness'][best_indices].copy()

    def optimize(self):
        """Run the IPSO optimization"""
        start_time = time.time()

        # Initialize personal bests and global bests
        for dim in range(self.D):
            swarm = self.swarms[dim]

            # Evaluate initial fitness
            for i in range(self.pop_size):
                # Set CP for evaluation
                temp_CP = self.CP.copy()
                temp_CP[dim] = swarm['positions'][i]

                fitness = self.calculate_fuzzy_entropy(temp_CP)
                swarm['pb_fitness'][i] = fitness
                swarm['pb_positions'][i] = swarm['positions'][i].copy()

                # Update global best for this dimension
                if fitness > swarm['gb_fitness']:
                    swarm['gb_fitness'] = fitness
                    swarm['gb_position'] = swarm['positions'][i].copy()

            # Update context parameter with this dimension's gb
            self.CP[dim] = swarm['gb_position']

        # Main optimization loop
        for self.iteration in range(self.max_iter):
            for dim in range(self.D):
                swarm = self.swarms[dim]

                # Check for stagnation
                if self.iteration > 0 and swarm['gb_fitness'] <= self.swarms[dim]['gb_fitness']:
                    swarm['stagnation_count'] += 1
                else:
                    swarm['stagnation_count'] = 0

                # Replace worst particles if stagnated
                if swarm['stagnation_count'] >= 5:  # arbitrary threshold
                    self.replace_worst_particles(dim)
                    swarm['stagnation_count'] = 0

                # Update velocity and position
                self.update_velocity_position(dim)

                # Evaluate new positions
                for i in range(self.pop_size):
                    # Set CP for evaluation
                    temp_CP = self.CP.copy()
                    temp_CP[dim] = swarm['positions'][i]

                    fitness = self.calculate_fuzzy_entropy(temp_CP)

                    # Update personal best
                    if fitness > swarm['pb_fitness'][i]:
                        swarm['pb_fitness'][i] = fitness
                        swarm['pb_positions'][i] = swarm['positions'][i].copy()

                        # Update global best
                        if fitness > swarm['gb_fitness']:
                            swarm['gb_fitness'] = fitness
                            swarm['gb_position'] = swarm['positions'][i].copy()

                # Update context parameter with this dimension's gb
                self.CP[dim] = swarm['gb_position']

            # Store best fitness
            current_best_fitness = max(swarm['gb_fitness'] for swarm in self.swarms)
            self.best_fitness_history.append(current_best_fitness)

            # Print progress
            if (self.iteration + 1) % 10 == 0:
                print(f"Iteration {self.iteration + 1}/{self.max_iter}, Best Fitness: {current_best_fitness:.4f}")

        # Get final thresholds
        thresholds = []
        for i in range(self.k_levels - 1):
            a = self.CP[2 * i]
            b = self.CP[2 * i + 1]
            thresholds.append((a + b) / 2)  # Eq. 7 in the paper

        # Sort thresholds
        thresholds = sorted(thresholds)

        # Add dummy thresholds
        thresholds = [0] + thresholds + [255]

        execution_time = time.time() - start_time
        print(f"Optimization completed in {execution_time:.2f} seconds")

        return thresholds, execution_time, self.best_fitness_history

    def segment_image(self, thresholds):
        """Segment image using the obtained thresholds"""
        # Convert L*a*b* to grayscale (using L* channel)
        gray = self.lab_image[:, :, 0]

        # Create segmented image
        segmented = np.zeros_like(gray, dtype=np.uint8)

        for i in range(len(thresholds) - 1):
            mask = (gray >= thresholds[i]) & (gray < thresholds[i + 1])
            segmented[mask] = i

        return segmented


def calculate_metrics(original, segmented, ground_truth, train_mask):
    """Calculate PSNR, SSIM, and classification metrics"""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    # Convert images to float and normalize
    original_float = original.astype(np.float64)
    segmented_float = segmented.astype(np.float64)

    # Calculate data range for PSNR
    data_range = original_float.max() - original_float.min()

    # Calculate PSNR with explicit data range
    psnr_value = psnr(original_float, segmented_float, data_range=data_range)

    # Calculate SSIM with explicit data range
    ssim_value = ssim(original_float, segmented_float,
                      data_range=data_range,
                      win_size=3)  # smaller window for smaller images

    # Prepare data for classification
    X = original.reshape(-1, 1)
    y = ground_truth.ravel()

    # Split into train and test (using provided train_mask)
    X_train = X[train_mask.ravel() == 1]
    y_train = y[train_mask.ravel() == 1]
    X_test = X[train_mask.ravel() == 0]
    y_test = y[train_mask.ravel() == 0]

    # Train SVM classifier
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    # Predict and calculate metrics
    y_pred = clf.predict(X_test)
    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    return psnr_value, ssim_value, oa, kappa


def load_indian_pines_data():
    """Load and preprocess Indian Pines dataset"""
    # Note: In a real implementation, you would load the actual dataset
    # For this example, we'll create a synthetic dataset of similar size

    # Create synthetic hyperspectral image (145x145x200)
    np.random.seed(42)
    image = np.random.randint(0, 256, (145, 145, 200), dtype=np.uint8)

    # Create synthetic ground truth (16 classes)
    ground_truth = np.zeros((145, 145), dtype=np.uint8)

    # Assign classes to different regions
    ground_truth[20:40, 30:60] = 1
    ground_truth[50:80, 10:40] = 2
    ground_truth[90:120, 70:100] = 3
    ground_truth[30:60, 80:110] = 4
    ground_truth[100:130, 20:50] = 5
    ground_truth[10:30, 100:130] = 6
    ground_truth[60:90, 50:80] = 7
    ground_truth[110:140, 90:120] = 8
    ground_truth[40:70, 120:140] = 9
    ground_truth[80:110, 30:60] = 10
    ground_truth[20:50, 50:80] = 11
    ground_truth[70:100, 90:120] = 12
    ground_truth[30:60, 10:40] = 13
    ground_truth[90:120, 40:70] = 14
    ground_truth[50:80, 70:100] = 15

    # Create training mask (10% of pixels)
    train_mask = np.zeros_like(ground_truth)
    for class_id in np.unique(ground_truth):
        if class_id == 0:  # skip background
            continue
        class_pixels = np.argwhere(ground_truth == class_id)
        n_train = max(1, int(0.1 * len(class_pixels)))  # 10% for training
        train_indices = np.random.choice(len(class_pixels), n_train, replace=False)
        for idx in train_indices:
            train_mask[class_pixels[idx][0], class_pixels[idx][1]] = 1

    return image, ground_truth, train_mask


def main():
    # Load Indian Pines dataset
    image, ground_truth, train_mask = load_indian_pines_data()

    # Reduce dimensionality with PCA (select first 3 components for visualization)
    pca = PCA(n_components=3)
    image_3d = pca.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape[0], image.shape[1], 3)

    # Normalize to 0-255 for display
    image_3d = (image_3d - image_3d.min()) / (image_3d.max() - image_3d.min()) * 255
    image_3d = image_3d.astype(np.uint8)

    # Run IPSO for different levels
    for k in [10, 12, 14]:
        print(f"\nRunning IPSO for {k} levels...")
        ipso = IPSO(image_3d, k_levels=k, population_size=50, max_iter=100)
        thresholds, exec_time, fitness_history = ipso.optimize()

        # Segment image
        segmented = ipso.segment_image(thresholds)

        # Calculate metrics
        psnr_val, ssim_val, oa, kappa = calculate_metrics(
            image_3d.mean(axis=2), segmented, ground_truth, train_mask
        )

        print(f"\nResults for {k} levels:")
        print(f"Thresholds: {thresholds}")
        print(f"PSNR: {psnr_val:.2f} dB")
        print(f"SSIM: {ssim_val:.4f}")
        print(f"OA: {oa:.4f}")
        print(f"Kappa: {kappa:.4f}")
        print(f"Execution Time: {exec_time:.2f} seconds")

        # Plot convergence
        plt.figure()
        plt.plot(fitness_history)
        plt.title(f'Convergence Plot for {k} Levels')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness (Fuzzy Entropy)')
        plt.show()

        # Display results
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(image_3d)
        plt.title('Original Image')
        plt.subplot(132)
        plt.imshow(ground_truth)
        plt.title('Ground Truth')
        plt.subplot(133)
        plt.imshow(segmented)
        plt.title(f'Segmented ({k} levels)')
        plt.show()


if __name__ == "__main__":
    main()