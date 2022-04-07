"""
The NetworkData and NetworkCollection classes for loading, saving, and representing neural networks and their weights and biases.
"""
import numpy as np
import os
import pickle
import tensorflow as tf

from dataclasses import dataclass
from network import Network


@dataclass
class NetworkData:
    """
    The NetworkData class holds the data for a single neural network.
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
    def load_from_original_subflow(base_data_folder: str, network_name: str, utilization: float) -> "NetworkData":
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
            layer = NetworkData.Layer(weight, weight_name, bias, bias_name)
            layers.append(layer)

        # Sanity check
        for layer in layers:
            assert layer.weight.dtype == np.float32
            assert layer.bias.dtype == np.float32

        return NetworkData(network_name, utilization, layers)

    @staticmethod
    def load_from_network(network: Network, utilization: float) -> "NetworkData":
        # Convert the data from the network to a NetworkData structure
        # todo: get better access to the network layers
        layers = list()
        for i, layer in enumerate(network._model.layers):
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                weights = layer.get_weights()[0]
                bias = layer.get_weights()[1]
                layers.append(NetworkData.Layer(weights, f"weight_{i}", bias, f"bias_{i}"))

        return NetworkData(network.name, utilization, layers)

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
    The NetworkCollection class holds a group of NetworkData instances.
    """

    networks: list[NetworkData]

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
        """
        Returns the number of networks in the collection.

        :return: The number of networks.
        """

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
    def weights_count(self) -> int:
        """
        Returns the number of weights in the collection.

        :return: The number of weights.
        """

        return self.combined_weights.size

    @property
    def biases_count(self) -> int:
        """
        Returns the number of biases in the collection.

        :return: The number of biases.
        """

        return self.combined_biases.size

    @property
    def parameter_count(self) -> int:
        """
        Returns the number of parameters in the collection.

        :return: The number of parameters.
        """

        return self.weights_count + self.biases_count

    @property
    def nonzero_weights_count(self) -> int:
        """
        Returns the number of non-zero weights in the collection.

        :return: The number of non-zero weights.
        """

        return np.count_nonzero(self.combined_weights)

    @property
    def nonzero_biases_count(self) -> int:
        """
        Returns the number of non-zero biases in the collection.

        :return: The number of non-zero biases.
        """

        return np.count_nonzero(self.combined_biases)

    @property
    def nonzero_parameter_count(self) -> int:
        """
        Returns the number of non-zero parameters in the collection.

        :return: The number of non-zero parameters.
        """

        return self.nonzero_weights_count + self.nonzero_biases_count

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

    @property
    def weight_magnitude_average(self) -> float:
        """
        Returns the average magnitude over all the weights of all networks.

        :return: The average weight magnitude.
        """

        flattened_weights = self.combined_weights.flatten()
        return float(np.mean(np.abs(flattened_weights)))

    @property
    def bias_magnitude_average(self) -> float:
        """
        Returns the average magnitude over all the biases of all networks.

        :return: The average bias magnitude.
        """

        flattened_biases = self.combined_biases.flatten()
        return float(np.mean(np.abs(flattened_biases)))

    @property
    def byte_size(self) -> int:
        """
        Returns the size of the whole network collection in bytes.

        :return: The size in bytes.
        """

        assert self.combined_weights.dtype == np.float32
        assert self.combined_biases.dtype == np.float32
        return self.parameter_count * 4

    @property
    def nonzero_byte_size(self) -> int:
        """
        Returns the size in bytes of the network collection counting only the non-zero parameters.

        :return: The size in bytes.
        """

        assert self.combined_weights.dtype == np.float32
        assert self.combined_biases.dtype == np.float32
        return self.nonzero_parameter_count * 4
