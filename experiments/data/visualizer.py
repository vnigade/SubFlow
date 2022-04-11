"""
Utility for displaying data examples.
"""
import matplotlib.pyplot as plt
import numpy as np


def display_examples(x_test: np.ndarray, y_test: np.ndarray, predictions: np.ndarray, indices: np.ndarray, examples_per_row: int = 5):
    assert indices.size == x_test.shape[0] == y_test.shape[0]
    count = indices.size
    rows = -1 * (-count // examples_per_row)
    cols = min(count, examples_per_row)

    figure, axes = plt.subplots(rows, cols, squeeze=False, tight_layout=True)
    for i, (index, x, y, prediction) in enumerate(zip(indices, x_test, y_test, predictions)):
        row, col = np.divmod(i, examples_per_row)
        axes[row, col].set_title(f"[{index}] class={y} / predicted={prediction}")
        axes[row, col].imshow(x, cmap="gray", vmin=0.0, vmax=1.0)
    plt.show()
    plt.close(figure)
