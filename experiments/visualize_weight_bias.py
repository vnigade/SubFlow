"""
Visualizes weights and biases that were extracted using the extract_data.py script.
"""
import argparse
import numpy as np

from network import NetworkCollection, Visualizer


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--collection_file", type=str, default="../output/collection.pickle", help="The network collection to load.")
    args = args.parse_args()

    # Load network collection
    collection = NetworkCollection.load_from_file(args.collection_file)
    print(collection)

    # Plot networks
    min_weight_value, max_weight_value = collection.weight_minmax
    min_bias_value, max_bias_value = collection.bias_minmax
    for network in collection.networks:
        output_file = f"visualization_{network.utilization:.2f}.png"
        Visualizer.plot_network(network, output_file, min_weight_value, max_weight_value, min_bias_value, max_bias_value, show_figure=False)

    # Compute and plot differences between weights
    combined_weights = collection.combined_weights
    weight_differences = np.diff(combined_weights, axis=1)
    weight_differences_percentage = (np.abs(weight_differences) / np.abs(combined_weights[:, 0].reshape(-1, 1))) * 100.0
    weight_differences_list = [weight_differences[:, i].flatten() for i in range(weight_differences.shape[1])]
    weight_differences_percentage_list = [weight_differences_percentage[:, i].flatten() for i in range(weight_differences_percentage.shape[1])]

    cutaway_ratio = 0.05
    combined_weights_quantiles = np.quantile(combined_weights, cutaway_ratio), np.quantile(combined_weights, 1.0 - cutaway_ratio)
    weight_differences_quantiles = np.quantile(weight_differences, cutaway_ratio), np.quantile(weight_differences, 1.0 - cutaway_ratio)
    weight_differences_percentage_quantiles = np.quantile(weight_differences_percentage, cutaway_ratio), np.quantile(weight_differences_percentage, 1.0 - cutaway_ratio)

    Visualizer.plot_arrays(weight_differences_list, "weight_differences.png", data_range=weight_differences_quantiles, show_figure=False)
    Visualizer.plot_arrays(weight_differences_percentage_list, "weight_differences_percentage.png", data_range=weight_differences_percentage_quantiles, show_figure=False)
    Visualizer.plot_histogram(combined_weights, "weight_histogram.png", data_range=combined_weights_quantiles)
    Visualizer.plot_histogram(weight_differences, "difference_histogram.png", data_range=weight_differences_quantiles)


if __name__ == "__main__":
    main()
