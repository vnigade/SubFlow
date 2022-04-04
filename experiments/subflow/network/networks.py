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

    def __init__(self, initialization_directory: Optional[str] = None, leaky_relu: bool = True):
        activation = "leaky_relu" if leaky_relu else "relu"
        layers = [tf.keras.layers.Input(shape=(28, 28, 1)),
                  tf.keras.layers.Conv2D(6, (5, 5), padding="valid", activation=activation),
                  tf.keras.layers.MaxPooling2D((2, 2)),
                  tf.keras.layers.Conv2D(16, (5, 5), padding="valid", activation=activation),
                  tf.keras.layers.MaxPooling2D((2, 2)),
                  tf.keras.layers.Flatten(),
                  tf.keras.layers.Dense(400, activation=activation),
                  tf.keras.layers.Dense(84, activation=activation),
                  tf.keras.layers.Dense(10),
                  tf.keras.layers.Softmax()]
        super(LeNet, self).__init__(self.__class__.__name__, layers, initialization_directory)


class SimpleLeNet(Network):
    """
    Implements a simpler LeNet architecture for testing.
    """

    def __init__(self, initialization_directory: Optional[str] = None, leaky_relu: bool = True):
        activation = "leaky_relu" if leaky_relu else "relu"
        layers = [tf.keras.layers.Input(shape=(28, 28, 1)),
                  tf.keras.layers.Flatten(),
                  tf.keras.layers.Dense(128, activation=activation),
                  tf.keras.layers.Dropout(0.2),
                  tf.keras.layers.Dense(10),
                  tf.keras.layers.Softmax()]
        super(SimpleLeNet, self).__init__(self.__class__.__name__, layers, initialization_directory)


# =================================================================================================
# SubFlow
# =================================================================================================

class SubFlow(Network):
    """
    Implements the LeNet model architecture with special SubFlow layers.
    """

    # =================================================================================================================================================================================================
    # Specialized Dense layer
    # =================================================================================================================================================================================================

    @tf.keras.utils.register_keras_serializable(package="SubFlow")
    class Dense(tf.keras.layers.Dense):
        def __init__(self, seed: int, utilization: int, units: int, **kwargs):
            super(SubFlow.Dense, self).__init__(units, **kwargs)

            activation_mask = self._create_activation_mask(seed, utilization, units)
            assert activation_mask.ndim == 1 and activation_mask.size == units
            float_mask = activation_mask.astype(np.float32)

            self.seed: int = seed
            self.utilization: int = utilization
            self.activation_mask: tf.Tensor = tf.constant(float_mask)
            self.active_neurons: int = np.count_nonzero(activation_mask)

        def call(self, inputs):
            output = super(SubFlow.Dense, self).call(inputs)
            result = tf.multiply(output, self.activation_mask)
            return result

        def get_config(self):
            config = super(SubFlow.Dense, self).get_config()
            config.update({"seed": self.seed, "utilization": self.utilization, "units": self.units})
            return config

        def active_neuron_count(self) -> int:
            return self.active_neurons

        @staticmethod
        def _create_activation_mask(seed: int, utilization: int, units: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            to_keep = int(float(units) * float(utilization) / 100.0)
            assert 0 <= to_keep <= units
            indices = np.arange(units)
            rng.shuffle(indices)
            indices = indices[:to_keep]
            mask = np.zeros(units, dtype=np.int8)
            mask[indices] = 1
            return mask

    # =================================================================================================================================================================================================
    # Specialized Conv2D layer
    # =================================================================================================================================================================================================

    @tf.keras.utils.register_keras_serializable(package="SubFlow")
    class Conv2D(tf.keras.layers.Conv2D):
        def __init__(self, seed: int, utilization: int, filters: int, kernel_size: tuple[int, int], **kwargs):
            super(SubFlow.Conv2D, self).__init__(filters, kernel_size, **kwargs)

            self.seed: int = seed
            self.utilization: int = utilization
            self.activation_mask: Optional[tf.Tensor] = None
            self.active_neurons: Optional[int] = None

        def build(self, input_shape):
            super(SubFlow.Conv2D, self).build(input_shape)

            # Compute output shape
            _, height, width, _ = input_shape
            resolution = np.array([height, width], dtype=np.int32)
            strides = np.array(self.strides, dtype=np.float32)
            kernel_size = np.array(self.kernel_size, dtype=np.float32)
            if self.padding == "valid":
                output_shape = np.ceil((resolution - kernel_size + 1).astype(np.float32) / strides).astype(np.int32)
            else:
                output_shape = np.ceil(resolution.astype(np.float32) / strides).astype(np.int32)

            # Create activation mask
            activation_mask = self._create_activation_mask(self.seed, self.utilization, output_shape, self.filters)
            assert activation_mask.ndim == 3
            float_mask = activation_mask.astype(np.float32)

            self.activation_mask = tf.constant(float_mask)
            self.active_neurons = np.count_nonzero(activation_mask)

        def call(self, inputs):
            assert self.activation_mask is not None
            output = super(SubFlow.Conv2D, self).call(inputs)
            result = tf.multiply(output, self.activation_mask)
            return result

        def get_config(self):
            config = super(SubFlow.Conv2D, self).get_config()
            config.update({"seed": self.seed, "utilization": self.utilization})
            return config

        def active_neuron_count(self) -> int:
            return self.active_neurons

        @staticmethod
        def _create_activation_mask(seed: int, utilization: int, output_shape: tuple[int, int], filter_count: int) -> np.ndarray:
            # note: this implementation tries to maintain the same ratio of utilization per feature map
            rng = np.random.default_rng(seed)
            per_feature_map_count = np.prod(output_shape)
            mask = np.zeros((*output_shape, filter_count), dtype=np.int8)

            for i in range(filter_count):
                to_keep = int(float(per_feature_map_count) * float(utilization) / 100.0)
                assert 0 <= to_keep <= per_feature_map_count
                indices = np.arange(per_feature_map_count)
                rng.shuffle(indices)
                indices = indices[:to_keep]
                feature_map_mask = np.zeros(per_feature_map_count, dtype=np.int8)
                feature_map_mask[indices] = 1
                mask[:, :, i] = feature_map_mask.reshape(output_shape)

            return mask

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def __init__(self, initialization_directory: Optional[str] = None, leaky_relu: bool = True, utilization: int = 100, seed: int = 123456789):
        assert 1 < utilization <= 100

        # Create activation masks
        rng = np.random.default_rng(seed)
        seeds = rng.integers(low=0, high=np.iinfo(np.int32).max, size=4)

        # Create layers
        activation = "leaky_relu" if leaky_relu else "relu"
        layers = [tf.keras.layers.Input(shape=(28, 28, 1)),
                  # todo: remove the need to pass the output shape to the layer (it can be derived in the build method).
                  SubFlow.Conv2D(seeds[0], utilization, 6, (5, 5), padding="valid", activation=activation),
                  tf.keras.layers.MaxPooling2D((2, 2)),
                  SubFlow.Conv2D(seeds[1], utilization, 16, (5, 5), padding="valid", activation=activation),
                  tf.keras.layers.MaxPooling2D((2, 2)),
                  tf.keras.layers.Flatten(),
                  SubFlow.Dense(seeds[2], utilization, 400, activation=activation),
                  SubFlow.Dense(seeds[3], utilization, 84, activation=activation),
                  tf.keras.layers.Dense(10),
                  tf.keras.layers.Softmax()]

        name = f"{self.__class__.__name__}_{utilization}"
        super(SubFlow, self).__init__(name, layers, initialization_directory)
