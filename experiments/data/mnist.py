"""
Provides utility for loading and sampling examples from the MNIST dataset.
"""
import numpy as np
import tensorflow as tf


def load_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def sample_examples(test_data: tuple[np.ndarray, np.ndarray], count: int = 20, seed: int = 123456789) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    x_test, y_test = test_data
    rng = np.random.default_rng(seed)
    indices = rng.integers(x_test.shape[0], size=count)
    return (x_test[indices], y_test[indices]), indices
