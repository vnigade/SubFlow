"""
Visualizes weights and biases that were extracted using the extract_data.py script.
"""
import argparse
import numpy as np
import os

from analysis import NetworkCollection, Visualizer


# =================================================================================================
# Helpers
# =================================================================================================

def array_columns_to_list(array: np.ndarray) -> list[np.ndarray]:
    _, cols = array.shape
    return [array[:, i].flatten() for i in range(cols)]


def quantile(array: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> tuple[float, float]:
    return np.quantile(array, lower), np.quantile(array, upper)


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--collection_file", type=str, default="./data/collection.pickle", help="The network collection to load.")
    args.add_argument("--plot_folder", type=str, default="./data", help="The folder for outputting the plots.")
    args = args.parse_args()

    # Load network collection
    collection = NetworkCollection.load_from_file(args.collection_file)
    print(collection)

    # Plot networks
    min_weight_value, max_weight_value = collection.weight_minmax
    min_bias_value, max_bias_value = collection.bias_minmax
    for network in collection.networks:
        output_file = os.path.join(args.plot_folder, f"visualization_{network.utilization:.2f}.png")
        Visualizer.plot_network(network, output_file, min_weight_value, max_weight_value, min_bias_value, max_bias_value, show_figure=False)

    # Compute differences and data quantiles
    combined_weights = collection.combined_weights
    base_weights = combined_weights[:, 0].reshape(-1, 1)
    weight_base_differences = combined_weights[:, 1:] - base_weights
    weight_previous_differences = np.diff(combined_weights, axis=1)
    weight_base_differences_ratio = np.abs(weight_base_differences) / np.abs(base_weights)
    weight_previous_differences_ratio = np.abs(weight_previous_differences) / np.abs(base_weights)

    # Plot weights data and histograms
    Visualizer.plot_arrays(array_columns_to_list(weight_base_differences), os.path.join(args.plot_folder, "weight_base_differences.png"), data_range=quantile(base_weights), show_figure=False)
    Visualizer.plot_arrays(array_columns_to_list(weight_previous_differences), os.path.join(args.plot_folder, "weight_previous_differences.png"), data_range=quantile(weight_previous_differences),
                           show_figure=False)
    Visualizer.plot_arrays(array_columns_to_list(weight_base_differences_ratio), os.path.join(args.plot_folder, "weight_base_differences_ratio.png"),
                           data_range=quantile(weight_base_differences_ratio), show_figure=False)
    Visualizer.plot_arrays(array_columns_to_list(weight_previous_differences_ratio), os.path.join(args.plot_folder, "weight_previous_differences_ratio.png"),
                           data_range=quantile(weight_previous_differences_ratio),
                           show_figure=False)

    Visualizer.plot_histogram(combined_weights, os.path.join(args.plot_folder, "combined_weights_histogram.png"), data_range=quantile(combined_weights))
    Visualizer.plot_histogram(weight_base_differences, os.path.join(args.plot_folder, "weight_base_differences_histogram.png"), data_range=quantile(weight_base_differences))
    Visualizer.plot_histogram(weight_previous_differences, os.path.join(args.plot_folder, "weight_previous_differences_histogram.png"), data_range=quantile(weight_previous_differences))


if __name__ == "__main__":
    main()
