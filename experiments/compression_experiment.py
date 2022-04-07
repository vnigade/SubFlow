"""
Experiments with weight compression.
"""
import argparse
import copy
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import sys
import tensorflow as tf

from analysis import NetworkCollection, NetworkData, NetworkDifference, NetworkDifferenceCollection
from dataclasses import dataclass
from network import SubFlow
from network.mnist import load_data


# =================================================================================================
# Network initialization
# =================================================================================================

def create_from_network_data(network_data: NetworkData, utilization: int, seed: int) -> SubFlow:
    """ Creates a SubFlow network from network data.
    todo: this function should be placed somewhere else.
    """

    network = SubFlow(None, True, utilization, seed)
    index = 0
    # todo: get better access to the network layers
    for layer in network._model.layers:
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            data_layer = network_data.layers[index]
            layer.set_weights([data_layer.weight, data_layer.bias])
            index += 1

    return network


# =================================================================================================
# Compression
# =================================================================================================

def compress_network_difference(network_difference: NetworkDifference, percentile: float) -> NetworkDifference:
    compressed = copy.deepcopy(network_difference)
    for layer in compressed.layers:
        layer.weight_difference[np.abs(layer.weight_difference) < percentile] = 0.0
    return compressed


def compress_differences_overall(differences: NetworkDifferenceCollection, compression_rate: int) -> NetworkDifferenceCollection:
    assert 0 <= compression_rate < 100
    # Find the percentile value for the compression rate over all weight differences of the difference collection
    # todo: also compress biases
    combined_weight_differences = differences.combined_weight_differences
    absolute_weight_differences = np.abs(combined_weight_differences)
    percentile = np.percentile(absolute_weight_differences, compression_rate)

    # Compress the whole difference collection using a single percentile value
    compressed_differences = list()
    for network_difference in differences.differences:
        compressed_network_difference = compress_network_difference(network_difference, percentile)
        compressed_differences.append(compressed_network_difference)

    return NetworkDifferenceCollection(compressed_differences)


def compress_differences_individual(differences: NetworkDifferenceCollection, compression_rate: int) -> NetworkDifferenceCollection:
    assert 0 <= compression_rate < 100
    # Compress the difference collection by using individual percentile values
    compressed_differences = list()
    for network_difference in differences.differences:
        # Find the percentile value for the compression rate for the network
        # todo: also compress biases
        network_weight_differences = network_difference.stacked_weight_differences
        absolute_weight_differences = np.abs(network_weight_differences)
        percentile = np.percentile(absolute_weight_differences, compression_rate)
        # Compress the network differences
        compressed_network_difference = compress_network_difference(network_difference, percentile)
        compressed_differences.append(compressed_network_difference)

    return NetworkDifferenceCollection(compressed_differences)


def compress_differences(differences: NetworkDifferenceCollection, compression_rate: int, individual: bool) -> NetworkDifferenceCollection:
    if individual:
        compressed_differences = compress_differences_individual(differences, compression_rate)
    else:
        compressed_differences = compress_differences_overall(differences, compression_rate)
    return compressed_differences


# =================================================================================================
# Evaluation
# =================================================================================================

@dataclass
class DifferenceStatistic:
    weight_differences_count: int
    nonzero_weight_differences_count: int
    bias_differences_count: int
    nonzero_bias_differences_count: int
    byte_size: int
    nonzero_byte_size: int

    def __init__(self, differences: NetworkDifferenceCollection):
        self.weight_differences_count = differences.weight_differences_count
        self.nonzero_weight_differences_count = differences.nonzero_weight_differences_count
        self.bias_differences_count = differences.bias_differences_count
        self.nonzero_bias_differences_count = differences.nonzero_bias_differences_count
        self.byte_size = differences.byte_size
        self.nonzero_byte_size = differences.nonzero_byte_size

    @property
    def nonzero_weight_differences_percentage(self) -> float:
        return float(self.nonzero_weight_differences_count) / float(self.weight_differences_count)

    @property
    def nonzero_bias_differences_percentage(self) -> float:
        return float(self.nonzero_bias_differences_count) / float(self.bias_differences_count)

    @property
    def nonzero_byte_size_percentage(self) -> float:
        return float(self.nonzero_byte_size) / float(self.byte_size)


def to_megabytes(x: int) -> float:
    return float(x) / (1024.0 ** 2)


def value_and_percentage(x: int, base: int) -> str:
    percentage = 100.0 * float(x) / float(base)
    return f"{x} ({percentage:.1f}%)"


def megabytes_and_percentage(x: int, base: int) -> str:
    megabytes = to_megabytes(x)
    percentage = 100.0 * float(x) / float(base)
    return f"{megabytes:.1f} MB ({percentage:.1f}%)"


def display_collection_statistics(collection: NetworkCollection) -> None:
    logging.info(f"Number of parameters: {collection.parameter_count}")
    logging.info(f"Number of non-zero parameters: {value_and_percentage(collection.nonzero_parameter_count, collection.parameter_count)}")
    logging.info(f"Average weight magnitude: {collection.weight_magnitude_average:.2f}")
    logging.info(f"Average bias magnitude: {collection.bias_magnitude_average:.2f}")
    logging.info(f"Full byte size: {to_megabytes(collection.byte_size):.1f} MB")
    logging.info(f"Non-zero byte size: {megabytes_and_percentage(collection.nonzero_byte_size, collection.byte_size)}")


def display_differences_statistics(statistic: DifferenceStatistic) -> None:
    logging.info(f"Number of non-zero weight differences: {statistic.nonzero_weight_differences_count} ({statistic.nonzero_weight_differences_percentage * 100.0:.1f}%)")
    logging.info(f"Number of non-zero bias differences: {statistic.nonzero_bias_differences_count} ({statistic.nonzero_bias_differences_percentage * 100.0:.1f}%)")
    logging.info(f"Differences byte size: {to_megabytes(statistic.byte_size):.1f} MB")
    logging.info(f"Non-zero differences byte size: {statistic.nonzero_byte_size} ({statistic.nonzero_byte_size_percentage * 100.0:.1f}%)")


def eval_base_differences(collection: NetworkCollection, differences: NetworkDifferenceCollection, x: np.ndarray, y: np.ndarray) -> pd.Series:
    assert collection.count - 1 == differences.count
    base_network_data = collection.networks[0]
    accuracies = list()
    for i in range(collection.count):
        # Restore network data from base network and differences
        network_data = copy.deepcopy(base_network_data)
        if i > 0:
            network_differences = differences.differences[i - 1]
            assert base_network_data.count == network_differences.count
            for k in range(base_network_data.count):
                network_data.layers[k].weight = base_network_data.layers[k].weight + network_differences.layers[k].weight_difference
                network_data.layers[k].bias = base_network_data.layers[k].bias + network_differences.layers[k].bias_difference

        # Restore the SubFlow network from the network data and evaluate it
        utilization = int(collection.networks[i].utilization * 100.0)
        network = create_from_network_data(network_data, utilization, seed=123456789)
        metrics = network.evaluate(x, y)
        accuracies.append(metrics["accuracy"])
        logging.info(f"utilization={utilization}: {metrics}")
    return pd.Series(accuracies)


def eval_previous_differences(collection: NetworkCollection, differences: NetworkDifferenceCollection, x: np.ndarray, y: np.ndarray) -> pd.Series:
    assert collection.count - 1 == differences.count
    network_data = copy.deepcopy(collection.networks[0])
    accuracies = list()
    for i in range(collection.count):
        # Create a next iteration of network data from the previous data and the network differences
        if i > 0:
            network_differences = differences.differences[i - 1]
            assert network_data.count == network_differences.count
            for k in range(network_data.count):
                network_data.layers[k].weight += network_differences.layers[k].weight_difference
                network_data.layers[k].bias += network_differences.layers[k].bias_difference

        # Restore the SubFlow network from the network data and evaluate it
        utilization = int(collection.networks[i].utilization * 100.0)
        network = create_from_network_data(network_data, utilization, seed=123456789)
        metrics = network.evaluate(x, y)
        accuracies.append(metrics["accuracy"])
        logging.info(f"utilization={utilization}: {metrics}")
    return pd.Series(accuracies)


def eval_differences(collection: NetworkCollection, differences: NetworkDifferenceCollection, x: np.ndarray, y: np.ndarray, relative_to_base: bool) -> pd.Series:
    if relative_to_base:
        return eval_base_differences(collection, differences, x, y)
    else:
        return eval_previous_differences(collection, differences, x, y)


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--collection_file", type=str, default="./data/from_lenet_leaky_5epochs.collection", help="The network collection to load.")
    args.add_argument("--output_base_directory", type=str, default="./results", help="The output log file.")
    args.add_argument("--relative_to_base", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Toggles the computation of the differences relative to the previous or base.")
    args.add_argument("--individual_compression", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Toggles overall or individual network compression.")
    args = args.parse_args()

    # Create the output base folder (if it not already exists)
    collection_file = pathlib.Path(args.collection_file)
    output_directory = os.path.join(args.output_base_directory, collection_file.stem)
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Define a base name for the experiment
    relative_to_base_str = "base" if args.relative_to_base else "previous"
    individual_compression_str = "individual" if args.individual_compression else "overall"
    base_name = f"{relative_to_base_str}_{individual_compression_str}"

    # Setup file logging
    log_file = os.path.join(output_directory, f"{base_name}_compression.log")
    logging.basicConfig(filename=log_file, filemode="w", level=logging.DEBUG, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Arguments: {vars(args)}")

    # Load MNIST dataset
    _, test = load_data()
    x_test, y_test = test
    logging.info(f"Test data: {x_test.shape} {y_test.shape}")
    logging.info("")

    # Load network collection
    collection = NetworkCollection.load_from_file(args.collection_file)
    logging.info(collection)
    display_collection_statistics(collection)
    logging.info("")

    # Compute network difference collection
    if args.relative_to_base:
        differences = NetworkDifferenceCollection.base_differences_from_collection(collection)
    else:
        differences = NetworkDifferenceCollection.previous_differences_from_collection(collection)

    logging.info("Original differences:")
    difference_statistic = DifferenceStatistic(differences)
    display_differences_statistics(difference_statistic)
    logging.info("")

    # Evaluate SubFlow with the original weights
    logging.info("=" * 150)
    logging.info("Original weights")
    logging.info("-" * 150)
    eval_differences(collection, differences, x_test, y_test, args.relative_to_base)
    logging.info("=" * 150)
    logging.info("")

    # Evaluate SubFlow with various compression rates
    all_accuracies = list()
    all_statistics = list()
    for compression_rate in range(10, 100, 10):
        compressed_differences = compress_differences(differences, compression_rate, args.individual_compression)
        compressed_difference_statistic = DifferenceStatistic(compressed_differences)
        all_statistics.append(compressed_difference_statistic)

        logging.info("=" * 150)
        logging.info(f"Compressed weights (rate = {compression_rate})")
        logging.info("-" * 150)
        display_differences_statistics(compressed_difference_statistic)

        # Track the accuracies for all utilization
        accuracies = eval_differences(collection, compressed_differences, x_test, y_test, args.relative_to_base)
        all_accuracies.append(accuracies)

        logging.info("-" * 150)
        logging.info("")

    # Write out difference statistics
    statistics_dict = {
        "weight_differences_count": pd.Series([s.weight_differences_count for s in all_statistics]),
        "nonzero_weight_differences_count": pd.Series([s.nonzero_weight_differences_count for s in all_statistics]),
        "bias_differences_count": pd.Series([s.bias_differences_count for s in all_statistics]),
        "nonzero_bias_differences_count": pd.Series([s.nonzero_bias_differences_count for s in all_statistics]),
        "byte_size": pd.Series([s.byte_size for s in all_statistics]),
        "nonzero_byte_size": pd.Series([s.nonzero_byte_size for s in all_statistics]),
        "nonzero_weight_differences_percentage": pd.Series([s.nonzero_weight_differences_percentage for s in all_statistics]),
        "nonzero_bias_differences_percentage": pd.Series([s.nonzero_bias_differences_percentage for s in all_statistics]),
        "nonzero_byte_size_percentage": pd.Series([s.nonzero_byte_size_percentage for s in all_statistics]),
    }
    statistics_file = os.path.join(output_directory, f"{base_name}_compression_statistics.xlsx")
    statistics_data_frame = pd.DataFrame(statistics_dict)
    statistics_data_frame.to_excel(statistics_file, index=False)

    # Write out evaluated accuracies over the various utilization
    accuracy_file = os.path.join(output_directory, f"{base_name}_compression_accuracy.xlsx")
    accuracy_data_frame = pd.DataFrame(all_accuracies).transpose()
    accuracy_data_frame.to_excel(accuracy_file, index=False)


if __name__ == "__main__":
    main()
