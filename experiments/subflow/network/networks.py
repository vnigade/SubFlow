"""
Implements network models for LeNet and SubFlow.
"""
import tensorflow as tf

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
                  tf.keras.layers.Dense(10, activation="relu")]
        super(LeNet, self).__init__(self.__class__.__name__, layers, checkpoint_directory)


class SimpleLeNet(Network):
    """
    Implements a simpler LeNet architecture for testing.
    """

    def __init__(self, checkpoint_directory: str):
        layers = [tf.keras.layers.Flatten(input_shape=(28, 28)),
                  tf.keras.layers.Dense(128, activation="relu"),
                  tf.keras.layers.Dropout(0.2),
                  tf.keras.layers.Dense(10)]
        super(SimpleLeNet, self).__init__(self.__class__.__name__, layers, checkpoint_directory)


# =================================================================================================
# SubFlow
# =================================================================================================

class SubFlow(Network):
    """
    Implements the LeNet model architecture with special SubFlow layers.
    """

    def __init__(self, checkpoint_directory: str):
        layers = []
        super(SubFlow, self).__init__(self.__class__.__name__, layers, checkpoint_directory)
