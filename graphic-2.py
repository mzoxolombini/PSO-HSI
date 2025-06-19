import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Sample effect size data (Cohen's d with 95% CI)
datasets = ['Indian Pines', 'PaviaU', 'Salinas']
methods = ['vs SVM-CK', 'vs EMAP+SVM', 'vs MRF-GMM']

effect_sizes = {
    'Indian Pines': {'d': [2.13, 1.87, 2.41], 'ci': [0.15, 0.18, 0.12]},
    'PaviaU': {'d': [1.95, 1.62, 2.23], 'ci': [0.17, 0.21, 0.14]},
    'Salinas': {'d': [2.53, 2.08, 2.76], 'ci': [0.13, 0.16, 0.11]}
}

# Create figure
plt.figure(figsize=(10, 6))
x = np.arange(len(datasets))  # dataset locations
width = 0.25  # width of bars
colors = ['#4e79a7', '#f28e2b', '#e15759']

# Plot effect sizes with error bars
for i, method in enumerate(methods):
    ds = [effect_sizes[ds]['d'][i] for ds in datasets]
    cis = [effect_sizes[ds]['ci'][i] for ds in datasets]
    plt.bar(x + i*width, ds, width, color=colors[i], label=method)
    plt.errorbar(x + i*width, ds, yerr=cis, fmt='none',
                 ecolor='black', capsize=5, capthick=1, elinewidth=1)

# Reference lines
plt.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.5)
plt.axhline(y=1.2, color='gray', linestyle='--', linewidth=0.5)
plt.axhline(y=2.0, color='gray', linestyle='--', linewidth=0.5)
plt.text(len(datasets)-0.7, 0.83, 'Large', fontsize=10, color='gray')
plt.text(len(datasets)-0.7, 1.23, 'Very Large', fontsize=10, color='gray')
plt.text(len(datasets)-0.7, 2.03, 'Huge', fontsize=10, color='gray')

# Formatting
plt.xticks(x + width, datasets)
plt.ylabel("Cohen's d Effect Size", fontsize=12)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=False, prune=None, nbins=6))
plt.ylim(0, 3.2)

# Title and legend
plt.title('Fig. 2. Standardised Effect Sizes for Accuracy Improvements',
          fontsize=14, pad=20)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),
           title='Comparison')

# Grid and borders
plt.grid(True, axis='y', linestyle=':', alpha=0.7)
for spine in ['top', 'right']:
    plt.gca().spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('effect_sizes.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('effect_sizes.png', dpi=300, bbox_inches='tight')
plt.show()