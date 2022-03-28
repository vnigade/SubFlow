"""
The Network and NetworkCollection classes for loading, saving, and representing neural networks and their weights and biases.
"""
import numpy as np
import os
import pickle

from dataclasses import dataclass


@dataclass
class Network:
    """
    The Network class holds the data for a single neural network.
    """

    @dataclass
    class Layer:
        weight: np.ndarray
        weight_name: str
        bias: np.ndarray
        bias_name: str

    # =================================================================================================================================================================================================
    # Members
    # =================================================================================================================================================================================================

    name: str
    utilization: float
    layers: list[Layer]

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    @staticmethod
    def load_from_folder(base_data_folder: str, network_name: str, utilization: float) -> "Network":
        # Get the weight and bias filenames
        data_folder = os.path.join(base_data_folder, network_name)
        files = os.listdir(data_folder)
        weight_files = [f for f in files if f.startswith("weight_")]
        bias_files = [f for f in files if f.startswith("bias_")]
        assert len(weight_files) == len(bias_files)

        # Create layers
        layers = list()
        for weight_file, bias_file in zip(weight_files, bias_files):
            weight = np.load(os.path.join(data_folder, weight_file))
            weight_name = os.path.splitext(weight_file)[0]
            bias = np.load(os.path.join(data_folder, bias_file))
            bias_name = os.path.splitext(bias_file)[0]
            layer = Network.Layer(weight, weight_name, bias, bias_name)
            layers.append(layer)

        # Sanity check
        for layer in layers:
            assert layer.weight.dtype == np.float32
            assert layer.bias.dtype == np.float32

        return Network(network_name, utilization, layers)

    def __repr__(self):
        return f"Network '{self.name}' [{self.utilization:.2f} utilization][{self.count} layers]"

    # =================================================================================================================================================================================================
    # Properties
    # =================================================================================================================================================================================================

    @property
    def count(self) -> int:
        return len(self.layers)

    @property
    def weight_count(self) -> int:
        return len(self.stacked_weights)

    @property
    def bias_count(self) -> int:
        return len(self.stacked_biases)

    @property
    def weight_names(self) -> list[str]:
        return [layer.weight_name for layer in self.layers]

    @property
    def bias_names(self) -> list[str]:
        return [layer.bias_name for layer in self.layers]

    @property
    def named_weights(self) -> dict[str, np.ndarray]:
        return {layer.weight_name: layer.weight for layer in self.layers}

    @property
    def named_bias(self) -> dict[str, np.ndarray]:
        return {layer.bias_name: layer.bias for layer in self.layers}

    @property
    def stacked_weights(self) -> np.ndarray:
        return np.concatenate([layer.weight.flatten() for layer in self.layers])

    @property
    def stacked_biases(self) -> np.ndarray:
        return np.concatenate([layer.bias.flatten() for layer in self.layers])

    @property
    def layer_weight_shapes(self) -> list[tuple]:
        return [layer.weight.shape for layer in self.layers]

    @property
    def layer_bias_shapes(self) -> list[tuple]:
        return [layer.bias.shape for layer in self.layers]

    @property
    def layer_weight_sizes(self) -> list[int]:
        return [layer.weight.size for layer in self.layers]

    @property
    def layer_bias_sizes(self) -> list[int]:
        return [layer.bias.size for layer in self.layers]

    @property
    def minimum_weight_size(self) -> int:
        return np.min(self.layer_weight_sizes)

    @property
    def maximum_weight_size(self) -> int:
        return np.max(self.layer_weight_sizes)

    @property
    def minimum_bias_size(self) -> int:
        return np.min(self.layer_bias_sizes)

    @property
    def maximum_bias_size(self) -> int:
        return np.max(self.layer_bias_sizes)

    @property
    def weight_minmax(self) -> tuple[float, float]:
        return np.min(self.stacked_weights), np.max(self.stacked_weights)

    @property
    def bias_minmax(self) -> tuple[float, float]:
        return np.min(self.stacked_biases), np.max(self.stacked_biases)


@dataclass
class NetworkCollection:
    """
    The NetworkCollection class holds a group of Networks.
    """

    networks: list[Network]

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def save(self, filename: str) -> None:
        file = open(filename, "wb")
        pickle.dump(self, file)

    @staticmethod
    def load_from_file(filename: str) -> "NetworkCollection":
        file = open(filename, "rb")
        collection = pickle.load(file)
        assert isinstance(collection, NetworkCollection)
        return collection

    def __repr__(self):
        return "\n".join([str(network) for network in self.networks])

    # =================================================================================================================================================================================================
    # Properties
    # =================================================================================================================================================================================================

    @property
    def count(self) -> int:
        return len(self.networks)

    @property
    def combined_weights(self) -> np.ndarray:
        """
        Combines all the weights from all the networks in one numpy array.
        The weights of each individual network are flattened into a single column vector.
        The column vectors for all networks are then stacked along the horizontal dimension.

        :return: Returns a numpy array (n x k) where k is the number of columns, one for each network and k is the total amount of weights.
        """

        return np.stack([network.stacked_weights for network in self.networks], axis=1)

    @property
    def combined_biases(self) -> np.ndarray:
        """
        Combines all the biases from all the networks in one numpy array.
        The biases of each individual network are flattened into a single column vector.
        The column vectors for all networks are then stacked along the horizontal dimension.

        :return: Returns a numpy array (n x k) where k is the number of columns, one for each network and k is the total amount of biases.
        """

        return np.stack([network.stacked_biases for network in self.networks], axis=1)

    @property
    def weight_minmax(self) -> tuple[float, float]:
        """
        Returns the min/max value over all the weights of all networks.

        :return: A tuple (min_value, max_value).
        """

        flattened_weights = self.combined_weights.flatten()
        return np.min(flattened_weights), np.max(flattened_weights)

    @property
    def bias_minmax(self) -> tuple[float, float]:
        """
        Returns the min/max value over all the biases of all networks.

        :return: A tuple (min_value, max_value).
        """

        flattened_biases = self.combined_biases.flatten()
        return np.min(flattened_biases), np.max(flattened_biases)
