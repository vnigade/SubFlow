"""
Sanity checks that de-activated parameters of SubFlow Dense layers are not altered during training.
"""
import argparse
import numpy as np
import tensorflow as tf

from data.mnist import load_data
from network import SubFlow


def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", type=int, default=1, help="The number of training epochs.")
    args = args.parse_args()

    # Load MNIST dataset
    train, test = load_data()
    x_train, y_train = train
    x_test, y_test = test

    # Create base SubFlow model and one with lower utilization
    base_network = SubFlow()
    lower_network = SubFlow(None, utilization=50)

    # Copy the parameters from the base to the lower utilized model
    # todo: get better access to the model layers
    for base_layer, lower_layer in zip(base_network._model.layers, lower_network._model.layers):
        if isinstance(base_layer, (SubFlow.Dense, SubFlow.Conv2D, tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            parameters = base_layer.get_weights()
            lower_layer.set_weights(parameters)

    # Compare that the weights are the same
    for base_layer, lower_layer in zip(base_network._model.layers, lower_network._model.layers):
        if isinstance(base_layer, (SubFlow.Dense, SubFlow.Conv2D, tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            base_weight, base_bias = base_layer.get_weights()
            lower_weight, lower_bias = lower_layer.get_weights()
            assert np.array_equal(base_weight, lower_weight)
            assert np.array_equal(base_bias, lower_bias)

    # Train the lower utilized model for a few epochs
    lower_network.train(None, x_train, y_train, args.epochs)

    # Sanity check that the weights in the Dense layers which are disabled by the activation mask are unaltered
    for base_layer, lower_layer in zip(base_network._model.layers, lower_network._model.layers):
        if isinstance(base_layer, SubFlow.Dense):
            base_weight, base_bias = base_layer.get_weights()
            lower_weight, lower_bias = lower_layer.get_weights()
            activation_mask = lower_layer.activation_mask.numpy()
            disabled_neurons = activation_mask == 0.0
            assert np.array_equal(base_weight[:, disabled_neurons], lower_weight[:, disabled_neurons])
            assert np.array_equal(base_bias[disabled_neurons], lower_bias[disabled_neurons])


if __name__ == "__main__":
    main()
