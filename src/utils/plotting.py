import matplotlib.pyplot as plt
import numpy as np
import sys
import os


sys.path.append(os.path.abspath("../../src"))

def plot_rawVStrans(arr_raw, arr_trans, column_names, output_dir=None, name=""):
    num_columns = arr_raw.shape[1]
    x = np.arange(num_columns)
    width = 0.4

    raw_means = np.mean(arr_raw, axis=0)
    raw_stds = np.std(arr_raw, axis=0)
    trans_means = np.mean(arr_trans, axis=0)
    trans_stds = np.std(arr_trans, axis=0)

    plt.figure(figsize=(15, 6))
    plt.bar(x - width / 2, raw_means, width, yerr=raw_stds, label='Raw', alpha=0.7, capsize=5)
    plt.bar(x + width / 2, trans_means, width, yerr=trans_stds, label='Transformed', alpha=0.7, capsize=5)
    plt.xticks(x, column_names, rotation=45, ha='right')
    plt.ylabel('Values')
    plt.title(f'Distribution: Raw vs. {name} Transformed')
    plt.legend()
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/distribution_comparison_{name}.png", bbox_inches='tight')
    else:
        plt.show()

