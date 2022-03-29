"""
Implements a base Network class.
"""
import numpy as np
import os
import tabulate
import tensorflow as tf

from tensorflow.python.keras.utils import layer_utils
from typing import Union


class Network:
    """
    The base Network class.
    """

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def __init__(self, name: str, layers: list[tf.keras.layers.Layer], checkpoint_directory: str):
        # Define members and their types
        self._name: str
        self._layers: list[tf.keras.layers.Layer]
        self._model: tf.keras.models.Sequential
        self._optimizer: str
        self._loss_fn: tf.keras.losses.Loss
        self._metrics: list[str]
        self._checkpoint_dir: str
        self._checkpoint_path: str
        self._checkpoint_callback: tf.keras.callbacks.ModelCheckpoint

        # Verify input
        self._assertInput(layers)

        # Initialize members
        self._name = name
        self._layers = layers
        self._optimizer = "adam"
        self._loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self._metrics = ["accuracy", tf.keras.metrics.SparseCategoricalAccuracy()]
        self._model = tf.keras.models.Sequential(layers, name=name)
        self._checkpoint_directory = os.path.join(checkpoint_directory, name)
        self._checkpoint_path = os.path.join(self._checkpoint_directory, "{epoch:04d}.checkpoint")
        self._checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self._checkpoint_path, save_weights_only=True, verbose=1)

        # Compile the model
        self._model.compile(optimizer=self._optimizer, loss=self._loss_fn, metrics=self._metrics)

    def __str__(self) -> str:
        layers = [layer for layer in self._layers if type(layer).__name__ != "KerasTensor"]
        dense_layers = [layer for layer in self._layers if isinstance(layer, tf.keras.layers.Dense)]
        conv_layers = [layer for layer in self._layers if isinstance(layer, tf.keras.layers.Conv2D)]

        values = list()
        for layer in layers:
            values.append([layer.name,
                           type(layer).__qualname__,
                           layer.input_shape,
                           layer.output_shape,
                           layer.count_params(),
                           self._weight_count(layer),
                           self._bias_count(layer),
                           self._neuron_count(layer),
                           self._active_neuron_count(layer),
                           f"{self._active_neuron_percentage(layer):.1f}%"])

        headers = ["Layer", "Type", "Input Shape", "Output Shape", "Param #", "Weight #", "Bias #", "Neuron #", "Active Neuron #", "Neuron %"]
        table = tabulate.tabulate(values, headers=headers)
        assert len(table) > 0

        trainable_count = layer_utils.count_params(self._model.trainable_weights)
        non_trainable_count = layer_utils.count_params(self._model.non_trainable_weights)
        total_neuron_count = np.sum([self._neuron_count(layer) for layer in layers])
        total_active_neuron_count = np.sum([self._active_neuron_count(layer) for layer in layers])
        neuron_percentage = 100.0 * float(total_active_neuron_count) / float(total_neuron_count)
        dense_neuron_count = np.sum([self._neuron_count(layer) for layer in dense_layers])
        conv_neuron_count = np.sum([self._neuron_count(layer) for layer in conv_layers])
        dense_active_neuron_count = np.sum([self._active_neuron_count(layer) for layer in dense_layers])
        conv_active_neuron_count = np.sum([self._active_neuron_count(layer) for layer in conv_layers])
        dense_conv_neuron_count = dense_neuron_count + conv_neuron_count
        dense_conv_active_neuron_count = dense_active_neuron_count + conv_active_neuron_count
        dense_conv_neuron_percentage = 100.0 * float(dense_active_neuron_count + conv_active_neuron_count) / float(dense_neuron_count + conv_neuron_count)
        summary = [f"Model: {self._model.name}",
                   f"Total params: {self._model.count_params()}",
                   f"Trainable params: {trainable_count}",
                   f"Non-trainable params: {non_trainable_count}",
                   f"Total neurons: {total_neuron_count} ({total_active_neuron_count} active, {neuron_percentage:.1f}%)",
                   f"Total neurons (Dense + Conv2D): {dense_conv_neuron_count} ({dense_conv_active_neuron_count} active, {dense_conv_neuron_percentage:.1f}%)"]

        separator = "=" * table.index("\n")
        return table + "\n" + separator + "\n" + "\n".join(summary)

    def summary(self) -> str:
        return str(self)

    def default_summary(self) -> str:
        lines = list()
        self._model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)

    def preload(self) -> None:
        latest = tf.train.latest_checkpoint(self._checkpoint_directory)
        if latest:
            self._model.load_weights(latest)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int) -> None:
        self._model.fit(x, y, epochs=epochs, callbacks=[self._checkpoint_callback])

    def evaluate(self, x, y) -> None:
        self._model.evaluate(x, y, verbose=2)

    def predict(self, x: np.ndarray, return_probabilities: bool = False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        probabilities = self._model.predict(x)
        predictions = np.argmax(probabilities, axis=1)
        if return_probabilities:
            return predictions, probabilities
        else:
            return predictions

    # =================================================================================================================================================================================================
    # Properties
    # =================================================================================================================================================================================================

    @property
    def name(self) -> str:
        raise self._name

    # =================================================================================================================================================================================================
    # Private methods
    # =================================================================================================================================================================================================

    @staticmethod
    def _assertInput(layers: list[tf.keras.layers.Layer]) -> None:
        if len(layers) < 2:
            raise RuntimeError("Need at least two layers.")
        if type(layers[0]).__name__ != "KerasTensor":
            raise RuntimeError("The first layer needs to be of type keras.Input.")
        if not isinstance(layers[-1], tf.keras.layers.Softmax):
            raise RuntimeError("The last layer must be of type keras.Softmax.")

    @staticmethod
    def _weight_count(layer: tf.keras.layers.Layer) -> int:
        count = 0
        if isinstance(layer, tf.keras.layers.Dense):
            count = np.prod([v for v in layer.output_shape if v]) * np.prod([v for v in layer.input_shape if v])
        elif isinstance(layer, tf.keras.layers.Conv2D):
            count = np.prod(layer.kernel_size) * layer.filters * layer.input_shape[-1]
        return count

    @staticmethod
    def _bias_count(layer: tf.keras.layers.Layer) -> int:
        count = 0
        if isinstance(layer, tf.keras.layers.Dense):
            count = np.prod([v for v in layer.output_shape if v])
        elif isinstance(layer, tf.keras.layers.Conv2D):
            count = layer.filters
        return count

    @staticmethod
    def _neuron_count(layer: tf.keras.layers.Layer) -> int:
        return int(np.prod([v for v in layer.output_shape if v]))

    @staticmethod
    def _active_neuron_count(layer: tf.keras.layers.Layer) -> int:
        # todo: this is a terrible design, fix it!
        method = getattr(layer, "active_neuron_count", None)
        if callable(method):
            count = layer.active_neuron_count()
        else:
            count = Network._neuron_count(layer)
        return count

    @staticmethod
    def _active_neuron_percentage(layer: tf.keras.layers.Layer) -> float:
        return 100.0 * float(Network._active_neuron_count(layer)) / float(Network._neuron_count(layer))
