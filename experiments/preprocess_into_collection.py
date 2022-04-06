"""
Collects the Network data extracted by the extract_from_subflow.py script and stores it into a NetworkCollection.
"""
import argparse
import os
import pathlib
import tensorflow as tf

from analysis import NetworkData, NetworkCollection
from network import BaseConfiguration, SubFlow, SubFlowLeNetConfiguration


# =================================================================================================
# Loading
# =================================================================================================

def load_from_original_subflow(base_data_folder: str) -> NetworkCollection:
    # Define input folders and utilization
    # todo: currently this is hard-coded for MNIST dataset and 10 utilizations and should be passed in as arguments.
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
    networks = [NetworkData.load_from_original_subflow(base_data_folder, name, utilization) for utilization, name in inputs.items()]
    return NetworkCollection(networks)


def load_from_models(base_model_folder: str, epochs: int) -> NetworkCollection:
    # Find and load configuration files in the folder that match the epochs
    configurations = dict()
    for config_file in pathlib.Path(base_model_folder).rglob("train.configuration"):
        # Load configuration file
        configuration = BaseConfiguration.load_from_file(str(config_file))
        assert isinstance(configuration, SubFlowLeNetConfiguration)
        if configuration.epochs == epochs:
            model_directory = str(config_file.parent)
            configurations[configuration.utilization] = (model_directory, configuration)

    # Load the networks in descending order of utilization
    networks = list()
    for utilization, (model_directory, configuration) in reversed(sorted(configurations.items())):
        # Load network
        network = SubFlow(None, configuration.leaky_relu, configuration.utilization, configuration.seed)
        network.load(model_directory)

        # Convert the data from the network to a NetworkData structure
        # todo: get better access to the network layers
        layers = list()
        for i, layer in enumerate(network._model.layers):
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                weights = layer.get_weights()[0]
                bias = layer.get_weights()[1]
                layers.append(NetworkData.Layer(weights, f"weight_{i}", bias, f"bias_{i}"))

        network_data = NetworkData(network.name, float(configuration.utilization) / 100.0, layers)
        networks.append(network_data)

    return NetworkCollection(networks)


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--base_data_folder", type=str, default="./models/SubFlow/Progressive", help="The original base data folder.")
    args.add_argument("--original_base_data_folder", type=str, default="../output/", help="The original base data folder.")
    args.add_argument("--epochs", type=int, default=5, help="The number of training epochs.")
    args.add_argument("--output_folder", type=str, default="./data", help="The output folder.")
    args.add_argument("--output_filename", type=str, default="collection.pickle", help="The output file name.")
    args = args.parse_args()

    # Load networks into collection
    # todo: currently all architectures are packed into one collection which is not desired.
    # collection = load_from_original_subflow(args.original_base_data_folder)
    collection = load_from_models(args.base_data_folder, args.epochs)
    print(collection)

    # Save collection
    output_filename = os.path.join(args.output_folder, args.output_filename)
    collection.save(output_filename)


if __name__ == "__main__":
    main()
