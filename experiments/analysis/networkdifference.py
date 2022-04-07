"""
The NetworkDifference and NetworkDifferenceCollection classes for representing neural network differences.
"""
import numpy as np
import pickle

from dataclasses import dataclass
from .networkdata import NetworkData, NetworkCollection


class NetworkDifference:
    """
    The NetworkDifference class holds the difference between two neural networks with the same architecture.
    """

    @dataclass
    class Layer:
        weight_difference: np.ndarray
        bias_difference: np.ndarray

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def __init__(self, network_a: NetworkData, network_b: NetworkData):
        # todo: add a check that the architectures of both networks are identical.
        self.layers: list[NetworkDifference.Layer] = list()
        for layer_a, layer_b in zip(network_a.layers, network_b.layers):
            layer = NetworkDifference.Layer(layer_b.weight - layer_a.weight, layer_b.bias - layer_a.bias)
            self.layers.append(layer)

    # =================================================================================================================================================================================================
    # Properties
    # =================================================================================================================================================================================================

    @property
    def count(self) -> int:
        return len(self.layers)

    @property
    def stacked_weight_differences(self) -> np.ndarray:
        return np.concatenate([layer.weight_difference.flatten() for layer in self.layers])

    @property
    def stacked_bias_differences(self) -> np.ndarray:
        return np.concatenate([layer.bias_difference.flatten() for layer in self.layers])

    @property
    def byte_size(self) -> int:
        assert self.stacked_weight_differences.dtype == np.float32
        assert self.stacked_bias_differences.dtype == np.float32
        return (self.stacked_weight_differences.size + self.stacked_bias_differences.size) * 4

    @property
    def nonzero_byte_size(self) -> int:
        assert self.stacked_weight_differences.dtype == np.float32
        assert self.stacked_bias_differences.dtype == np.float32
        nonzero_weight_count = np.count_nonzero(self.stacked_weight_differences)
        nonzero_bias_count = np.count_nonzero(self.stacked_bias_differences)
        return (nonzero_weight_count + nonzero_bias_count) * 4


@dataclass
class NetworkDifferenceCollection:
    """
    The NetworkDifferenceCollection class holds a collection of NetworkDifferences.
    """

    differences: list[NetworkDifference]

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    @staticmethod
    def base_differences_from_collection(collection: NetworkCollection) -> "NetworkDifferenceCollection":
        base_network = collection.networks[0]
        differences = list()
        for network in collection.networks[1:]:
            difference = NetworkDifference(base_network, network)
            differences.append(difference)

        return NetworkDifferenceCollection(differences)

    @staticmethod
    def previous_differences_from_collection(collection: NetworkCollection) -> "NetworkDifferenceCollection":
        differences = list()
        for i in range(1, collection.count):
            difference = NetworkDifference(collection.networks[i - 1], collection.networks[i])
            differences.append(difference)

        return NetworkDifferenceCollection(differences)

    def save(self, filename: str) -> None:
        file = open(filename, "wb")
        pickle.dump(self, file)

    @staticmethod
    def load_from_file(filename: str) -> "NetworkDifferenceCollection":
        file = open(filename, "rb")
        collection = pickle.load(file)
        assert isinstance(collection, NetworkDifferenceCollection)
        return collection

    def __repr__(self):
        return "\n".join([str(difference) for difference in self.differences])

    # =================================================================================================================================================================================================
    # Properties
    # =================================================================================================================================================================================================

    @property
    def count(self) -> int:
        return len(self.differences)

    @property
    def combined_weight_differences(self) -> np.ndarray:
        return np.stack([difference.stacked_weight_differences for difference in self.differences], axis=1)

    @property
    def combined_bias_differences(self) -> np.ndarray:
        return np.stack([difference.stacked_bias_differences for difference in self.differences], axis=1)

    @property
    def weight_differences_count(self) -> int:
        return self.combined_weight_differences.size

    @property
    def bias_differences_count(self) -> int:
        return self.combined_bias_differences.size

    @property
    def nonzero_weight_differences_count(self) -> int:
        return np.count_nonzero(self.combined_weight_differences)

    @property
    def nonzero_bias_differences_count(self) -> int:
        return np.count_nonzero(self.combined_bias_differences)

    @property
    def weight_differences_minmax(self) -> tuple[float, float]:
        flattened_weight_differences = self.combined_weight_differences.flatten()
        return np.min(flattened_weight_differences), np.max(flattened_weight_differences)

    @property
    def bias_differences_minmax(self) -> tuple[float, float]:
        flattened_bias_differences = self.combined_bias_differences.flatten()
        return np.min(flattened_bias_differences), np.max(flattened_bias_differences)

    @property
    def weight_differences_magnitude_average(self) -> float:
        flattened_weight_differences = self.combined_weight_differences.flatten()
        return float(np.mean(np.abs(flattened_weight_differences)))

    @property
    def bias_differences_magnitude_average(self) -> float:
        flattened_bias_differences = self.combined_bias_differences.flatten()
        return float(np.mean(np.abs(flattened_bias_differences)))

    @property
    def byte_size(self) -> int:
        """
        Returns the size of the whole network difference collection in bytes.

        :return: The size in bytes.
        """

        assert self.combined_weight_differences.dtype == np.float32
        assert self.combined_bias_differences.dtype == np.float32
        return (self.weight_differences_count + self.bias_differences_count) * 4

    @property
    def nonzero_byte_size(self) -> int:
        """
        Returns the size in bytes of the network difference collection counting only the non-zero differences.

        :return: The size in bytes.
        """

        assert self.combined_weight_differences.dtype == np.float32
        assert self.combined_bias_differences.dtype == np.float32
        return (self.nonzero_weight_differences_count + self.nonzero_bias_differences_count) * 4
