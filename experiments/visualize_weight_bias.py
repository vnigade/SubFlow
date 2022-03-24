"""
Visualizes weights and biases that were extracted using the extract_data.py script.
"""
import argparse
import numpy as np

from network import Network, NetworkCollection, Visualizer


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--base_data_folder", type=str, default="../output/", help="The base data folder.")
    args = args.parse_args()

    # Define input folders and utilization
    # todo: this should be passed in as arguments.
    inputs = {1.0: "network1",
              0.9: "sub_network9",
              0.8: "sub_network8",
              0.7: "sub_network7",
              0.6: "sub_network6",
              0.5: "sub_network5",
              0.4: "sub_network4",
              0.3: "sub_network3",
              0.2: "sub_network2",
              0.1: "sub_network1"}

    # Load networks
    networks = [Network.load_from_folder(args.base_data_folder, name, utilization) for utilization, name in inputs.items()]
    collection = NetworkCollection(networks)
    print(collection)

    # Plot networks
    # min_weight_value, max_weight_value = collection.weight_minmax
    # min_bias_value, max_bias_value = collection.bias_minmax
    # for network in networks:
    #     output_file = f"visualization_{network.utilization:.2f}.png"
    #     Visualizer.plot_network(network, output_file, min_weight_value, max_weight_value, min_bias_value, max_bias_value, show_figure=False)

    # Compute and plot differences between weights
    combined_weights = collection.combined_weights
    weight_differences = np.diff(combined_weights, axis=1)
    weight_relative_differences = weight_differences / combined_weights[:, 0].reshape(-1, 1)

    weight_differences_list = [weight_differences[:, i].flatten() for i in range(weight_differences.shape[1])]
    weight_relative_differences_list = [weight_relative_differences[:, i].flatten() for i in range(weight_relative_differences.shape[1])]
    Visualizer.plot_arrays(weight_differences_list, "weight_differences.png", same_range=True, show_figure=False)
    Visualizer.plot_arrays(weight_relative_differences_list, "weight_differences_percentage.png", same_range=True, show_figure=False)


if __name__ == "__main__":
    main()
