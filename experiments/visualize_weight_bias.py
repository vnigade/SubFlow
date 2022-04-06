"""
Visualizes weights and biases that were extracted using the extract_data.py script.
"""
import argparse
import numpy as np
import os
import pathlib

from analysis import NetworkCollection, Visualizer


# =================================================================================================
# Helpers
# =================================================================================================

def array_columns_to_list(array: np.ndarray) -> list[np.ndarray]:
    _, cols = array.shape
    return [array[:, i].flatten() for i in range(cols)]


def quantile(array: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> tuple[float, float]:
    return np.quantile(array, lower), np.quantile(array, upper)


def plot_networks(output_path: str, collection: NetworkCollection) -> None:
    min_weight_value, max_weight_value = collection.weight_minmax
    min_bias_value, max_bias_value = collection.bias_minmax
    for network in collection.networks:
        output_file = os.path.join(output_path, f"network_{network.utilization:.2f}_image.png")
        Visualizer.plot_network(network, output_file, min_weight_value, max_weight_value, min_bias_value, max_bias_value, show_figure=False)


def plot_network_differences(output_path: str, differences: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    difference_to_base, difference_to_previous, difference_ratio_to_base, difference_ratio_to_previous = differences
    Visualizer.plot_arrays(array_columns_to_list(difference_to_base), os.path.join(output_path, "difference_to_base.png"), data_range=None)
    Visualizer.plot_arrays(array_columns_to_list(difference_to_previous), os.path.join(output_path, "difference_to_previous.png"), data_range=None)
    Visualizer.plot_arrays(array_columns_to_list(difference_ratio_to_base), os.path.join(output_path, "difference_ratio_to_base.png"), data_range=None)
    Visualizer.plot_arrays(array_columns_to_list(difference_ratio_to_previous), os.path.join(output_path, "difference_ratio_to_previous.png"), data_range=None)


def plot_weight_histograms(output_path: str, combined_weights: np.ndarray, difference_to_base: np.ndarray, difference_to_previous: np.ndarray) -> None:
    Visualizer.plot_histogram(combined_weights, os.path.join(output_path, "histogram_combined_weights.png"), data_range=None)
    Visualizer.plot_histogram(difference_to_base, os.path.join(output_path, "histogram_difference_to_base.png"), data_range=None)
    Visualizer.plot_histogram(difference_to_previous, os.path.join(output_path, "histogram_difference_to_previous.png"), data_range=None)


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--collection_file", type=str, default="./data/from_scratch_collection.pickle", help="The network collection to load.")
    args.add_argument("--plot_base_folder", type=str, default="./data", help="The folder for outputting the plots.")
    args = args.parse_args()

    # Load network collection
    collection = NetworkCollection.load_from_file(args.collection_file)
    print(collection)

    # Compute weight differences
    combined_weights = collection.combined_weights
    base_weights = combined_weights[:, 0].reshape(-1, 1)
    difference_to_base = combined_weights[:, 1:] - base_weights
    difference_to_previous = np.diff(combined_weights, axis=1)
    difference_ratio_to_base = np.abs(difference_to_base) / np.abs(base_weights)
    difference_ratio_to_previous = np.abs(difference_to_previous) / np.abs(combined_weights[:, 0:-1])

    # Create plot output folder
    collection_path = pathlib.Path(args.collection_file)
    plot_folder = os.path.join(args.plot_base_folder, collection_path.stem)
    pathlib.Path(plot_folder).mkdir(parents=True, exist_ok=True)

    # Plot networks, weight histograms, and network differences
    plot_networks(plot_folder, collection)
    plot_network_differences(plot_folder, (difference_to_base, difference_to_previous, difference_ratio_to_base, difference_ratio_to_previous))
    plot_weight_histograms(plot_folder, combined_weights, difference_to_base, difference_to_previous)


if __name__ == "__main__":
    main()
