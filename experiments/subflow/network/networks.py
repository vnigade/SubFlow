"""
Implements network models for LeNet and SubFlow.
"""
import numpy as np
import tensorflow as tf

from typing import Optional
from .network import Network


# =================================================================================================
# LeNet
# =================================================================================================

class LeNet(Network):
    """
    Implements the LeNet model architecture as it is used in SubFlow.
    """

    def __init__(self, checkpoint_directory: str):
        layers = [tf.keras.layers.Input(shape=(28, 28, 1)),
                  tf.keras.layers.Conv2D(6, (5, 5), padding="valid", activation="relu"),
                  tf.keras.layers.MaxPooling2D((2, 2)),
                  tf.keras.layers.Conv2D(16, (5, 5), padding="valid", activation="relu"),
                  tf.keras.layers.MaxPooling2D((2, 2)),
                  tf.keras.layers.Flatten(),
                  tf.keras.layers.Dense(400, activation="relu"),
                  tf.keras.layers.Dense(84, activation="relu"),
                  tf.keras.layers.Dense(10, activation="relu"),
                  tf.keras.layers.Softmax()]
        super(LeNet, self).__init__(self.__class__.__name__, layers, checkpoint_directory)


class SimpleLeNet(Network):
    """
    Implements a simpler LeNet architecture for testing.
    """

    def __init__(self, checkpoint_directory: str):
        layers = [tf.keras.layers.Input(shape=(28, 28, 1)),
                  tf.keras.layers.Flatten(),
                  tf.keras.layers.Dense(128, activation="relu"),
                  tf.keras.layers.Dropout(0.2),
                  tf.keras.layers.Dense(10),
                  tf.keras.layers.Softmax()]
        super(SimpleLeNet, self).__init__(self.__class__.__name__, layers, checkpoint_directory)


# =================================================================================================
# SubFlow
# =================================================================================================

class SubFlow(Network):
    """
    Implements the LeNet model architecture with special SubFlow layers.
    """

    # =================================================================================================================================================================================================
    # Specialized layers
    # =================================================================================================================================================================================================

    class Dense(tf.keras.layers.Dense):
        def __init__(self, activation_mask: np.ndarray, units, activation=None):
            super(SubFlow.Dense, self).__init__(units, activation)
            assert activation_mask.ndim == 1 and activation_mask.size == units
            self.active_neurons = np.count_nonzero(activation_mask)
            float_mask = activation_mask.astype(np.float32)
            self.activation_mask: tf.Tensor = tf.constant(float_mask)

        def call(self, inputs):
            output = super(SubFlow.Dense, self).call(inputs)
            result = tf.multiply(output, self.activation_mask)
            return result

        def active_neuron_count(self) -> int:
            return self.active_neurons

    class Conv2D(tf.keras.layers.Conv2D):
        def __init__(self, activation_mask: np.ndarray, filters: int, kernel_size: tuple[int, int], padding=None, activation=None):
            super(SubFlow.Conv2D, self).__init__(filters, kernel_size, padding=padding, activation=activation)
            assert activation_mask.ndim == 3
            self.active_neurons = np.count_nonzero(activation_mask)
            float_mask = activation_mask.astype(np.float32)
            self.activation_mask: tf.Tensor = tf.constant(float_mask)

        def call(self, inputs):
            output = super(SubFlow.Conv2D, self).call(inputs)
            result = tf.multiply(output, self.activation_mask)
            return result

        def active_neuron_count(self) -> int:
            return self.active_neurons

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def __init__(self, checkpoint_directory: str, utilization: int, seed: int):
        assert 1 < utilization <= 100

        # Create activation masks
        rng = np.random.default_rng(seed)
        conv2d_activation_mask0 = self.get_conv2d_activation_mask(rng, 6, (24, 24), utilization)
        conv2d_activation_mask1 = self.get_conv2d_activation_mask(rng, 16, (8, 8), utilization)
        dense_activation_mask0 = self.get_dense_activation_mask(rng, 400, utilization)
        dense_activation_mask1 = self.get_dense_activation_mask(rng, 84, utilization)

        # Create layers
        layers = [tf.keras.layers.Input(shape=(28, 28, 1)),
                  SubFlow.Conv2D(conv2d_activation_mask0, 6, (5, 5), padding="valid", activation="relu"),
                  tf.keras.layers.MaxPooling2D((2, 2)),
                  SubFlow.Conv2D(conv2d_activation_mask1, 16, (5, 5), padding="valid", activation="relu"),
                  tf.keras.layers.MaxPooling2D((2, 2)),
                  tf.keras.layers.Flatten(),
                  SubFlow.Dense(dense_activation_mask0, 400, activation="relu"),
                  SubFlow.Dense(dense_activation_mask1, 84, activation="relu"),
                  tf.keras.layers.Dense(10, activation="relu"),
                  tf.keras.layers.Softmax()]

        name = f"{self.__class__.__name__}_{utilization}"
        super(SubFlow, self).__init__(name, layers, checkpoint_directory)

    # =================================================================================================================================================================================================
    # Private methods
    # =================================================================================================================================================================================================

    def get_dense_activation_mask(self, rng: np.random.Generator, count: int, utilization: int) -> np.ndarray:
        to_keep = int(float(count) * float(utilization) / 100.0)
        assert 0 <= to_keep <= count
        indices = np.arange(count)
        rng.shuffle(indices)
        indices = indices[:to_keep]
        mask = np.zeros(count, dtype=np.int8)
        mask[indices] = 1
        return mask

    def get_conv2d_activation_mask(self, rng: np.random.Generator, filter_count: int, input_shape: tuple[int, int], utilization: int) -> np.ndarray:
        # note: this implementation tries to maintain the same ratio of utilization per feature map
        per_feature_map_count = np.prod(input_shape)
        mask = np.zeros((*input_shape, filter_count), dtype=np.int8)

        for i in range(filter_count):
            to_keep = int(float(per_feature_map_count) * float(utilization) / 100.0)
            assert 0 <= to_keep <= per_feature_map_count
            indices = np.arange(per_feature_map_count)
            rng.shuffle(indices)
            indices = indices[:to_keep]
            feature_map_mask = np.zeros(per_feature_map_count, dtype=np.int8)
            feature_map_mask[indices] = 1
            mask[:, :, i] = feature_map_mask.reshape(input_shape)

        return mask
