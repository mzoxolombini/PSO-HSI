import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sample data (replace with your actual results)
methods = ['SVM-CK', 'EMAP+SVM', 'MRF-GMM', 'PSO-RL-FE (Ours)']
datasets = ['Indian Pines', 'PaviaU', 'Salinas']

# OA results with 95% CI (mean ± margin_of_error)
data = {
    'SVM-CK': {'mean': [83.5, 85.7, 87.2], 'ci': [1.2, 1.1, 0.9]},
    'EMAP+SVM': {'mean': [85.2, 88.3, 89.1], 'ci': [0.9, 0.8, 0.7]},
    'MRF-GMM': {'mean': [81.7, 84.2, 86.5], 'ci': [1.5, 1.3, 1.0]},
    'PSO-RL-FE (Ours)': {'mean': [89.8, 91.4, 93.5], 'ci': [0.7, 0.6, 0.5]}
}

# Create figure
plt.figure(figsize=(12, 6))
x = np.arange(len(datasets))  # dataset locations
width = 0.2  # width of bars
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot bars with error caps
for i, method in enumerate(methods):
    means = data[method]['mean']
    cis = data[method]['ci']
    plt.bar(x + i*width, means, width, color=colors[i], label=method)
    plt.errorbar(x + i*width, means, yerr=cis, fmt='none',
                 ecolor='black', capsize=5, capthick=1)

# Formatting
plt.xticks(x + width*1.5, datasets)
plt.ylabel('Overall Accuracy (%)', fontsize=12)
plt.title('Fig. 1. Overall Accuracy Comparison Across Methods and Datasets',
          fontsize=14, pad=20)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add significance markers
sig_y = 95
for i in range(len(datasets)):
    plt.plot([x[i] + 3*width, x[i] + 3*width, x[i] + 2*width, x[i] + 2*width],
             [sig_y, sig_y+1, sig_y+1, sig_y], lw=1, c='k')
    plt.text(x[i] + 2.5*width, sig_y+1.5, '***', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('Fig1_OA_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()