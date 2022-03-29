"""
Trains a standard Google LeNet on the MNIST dataset.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from network import Network, LeNet, SimpleLeNet


# =================================================================================================
# Functions
# =================================================================================================

def load_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def train_network(network: Network, train: tuple[np.ndarray, np.ndarray], epochs: int = 5):
    x_train, y_train = train
    network.train(x_train, y_train, epochs)


def eval_network(network: Network, test: tuple[np.ndarray, np.ndarray]):
    x_test, y_test = test
    network.evaluate(x_test, y_test)


def display_examples(network: Network, test: tuple[np.ndarray, np.ndarray], count: int = 20, examples_per_row: int = 5, seed: int = 123456789):
    x_test, y_test = test
    rng = np.random.default_rng(seed)
    indices = rng.integers(x_test.shape[0], size=count)
    examples = zip(x_test[indices], y_test[indices])
    predictions = network.predict(x_test[indices])

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
    args.add_argument("--checkpoint_path", type=str, default="./checkpoints", help="The network checkpoint path.")
    args = args.parse_args()

    # Load MNIST
    train, test = load_data()
    (x_train, y_train), (x_test, y_test) = train, test
    print(f"Train data: {x_train.shape} {y_train.shape}")
    print(f"Test data: {x_test.shape} {y_test.shape}")

    # Create network, then train and eval
    network = LeNet(args.checkpoint_path)
    # network = SimpleLeNet(args.checkpoint_path)
    print(network)
    print()

    # network.preload()
    train_network(network, train, 3)
    eval_network(network, test)

    # Display some examples
    display_examples(network, test)


if __name__ == "__main__":
    main()
