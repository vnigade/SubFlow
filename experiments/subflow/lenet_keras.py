"""
Trains a standard Google LeNet on the MNIST dataset.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from typing import Callable


# =================================================================================================
# Model
# =================================================================================================

class Model:
    def __init__(self, checkpoint_path: str):
        self._optimizer: str = "adam"
        self._loss_fn: tf.keras.losses.SparseCategoricalCrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._metrics = ["accuracy"]

        self._model: tf.keras.models.Sequential = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        self._model.compile(optimizer=self._optimizer, loss=self._loss_fn, metrics=self._metrics)
        self._probability_model: tf.keras.Sequential = tf.keras.Sequential([self._model, tf.keras.layers.Softmax()])

        self._checkpoint_dir: str = checkpoint_path
        self._checkpoint_path: str = os.path.join(checkpoint_path, "{epoch:04d}.checkpoint")
        self._checkpoint_callback: tf.keras.callbacks.ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self._checkpoint_path, save_weights_only=True, verbose=1)

    def __str__(self) -> str:
        summary = list()
        self._model.summary(print_fn=lambda x: summary.append(x))
        return "\n".join(summary)

    def preload(self) -> None:
        latest = tf.train.latest_checkpoint(self._checkpoint_dir)
        if latest:
            self._model.load_weights(latest)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int) -> None:
        self._model.fit(x, y, epochs=epochs, callbacks=[self._checkpoint_callback])

    def evaluate(self, x, y) -> None:
        self._model.evaluate(x, y, verbose=2)

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self._model(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self._model(x), axis=1)

    def eval_loss(self, predictions: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        return self._loss_fn(ground_truth, predictions)

    def probability(self, x: np.ndarray) -> np.ndarray:
        return self._probability_model(x)


# =================================================================================================
# Functions
# =================================================================================================

def load_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def train_model(model: Model, train: tuple[np.ndarray, np.ndarray], epochs: int = 5):
    x_train, y_train = train
    model.train(x_train, y_train, epochs)


def eval_model(model: Model, test: tuple[np.ndarray, np.ndarray]):
    x_test, y_test = test
    model.evaluate(x_test, y_test)


def display_examples(model: Model, test: tuple[np.ndarray, np.ndarray], count: int = 20, examples_per_row: int = 5, seed: int = 123456789):
    x_test, y_test = test
    rng = np.random.default_rng(seed)
    indices = rng.integers(x_test.shape[0], size=count)
    examples = zip(x_test[indices], y_test[indices])
    predictions = model.predict(x_test[indices])

    rows = -1 * -count // examples_per_row
    cols = min(count, examples_per_row)
    figure, axes = plt.subplots(rows, cols, tight_layout=True)
    for i, (index, (x, y), prediction) in enumerate(zip(indices, examples, predictions)):
        row, col = np.divmod(i, examples_per_row)
        axes[row, col].set_title(f"[{index}] class={y} / predicted={prediction}")
        axes[row, col].imshow(x, cmap="gray", vmin=0.0, vmax=1.0)
    plt.show()


# =================================================================================================
# Main
# =================================================================================================

def main():
    print("TensorFlow version:", tf.__version__)

    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint_path", type=str, default="./checkpoints/", help="The model checkpoint path.")
    args = args.parse_args()

    # Load MNIST
    train, test = load_data()
    (x_train, y_train), (x_test, y_test) = train, test
    print(f"Train data: {x_train.shape} {y_train.shape}")
    print(f"Test data: {x_test.shape} {y_test.shape}")

    # Train and eval
    model = Model(args.checkpoint_path)
    print(model)
    model.preload()
    # train_model(model, train, 3)
    eval_model(model, test)

    # Display some examples
    display_examples(model, test)


if __name__ == "__main__":
    main()
