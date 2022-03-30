"""
Trains a standard Google LeNet on the MNIST dataset.
"""
import argparse
import enum
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from network import Network, LeNet, SimpleLeNet, SubFlow


# =================================================================================================
# Functions
# =================================================================================================

def load_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


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

    # Define supported networks
    class Networks(enum.Enum):
        LENET = "lenet"
        SIMPLE_LENET = "simplelenet"
        SUBFLOW = "subflow"

    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint_path", type=str, default="./checkpoints", help="The network checkpoint path.")
    args.add_argument("--network", type=str, default="subflow", choices=[n.value for n in Networks], help="The network to run.")
    args.add_argument("--epochs", type=int, default=5, help="The number of training epochs.")
    args.add_argument("--utilization", type=int, default=20, help="The network utilization in percentage (integer values 1 to 100) [SubFlow only].")
    args.add_argument("--seed", type=int, default=123456789, help="The random seed for activation mask sampling [SubFlow only].")
    args.add_argument("--display_examples", type=bool, default=False, help="If True, displays some examples using MATPLOTLIB.")
    args = args.parse_args()

    # Load MNIST
    train, test = load_data()
    (x_train, y_train), (x_test, y_test) = train, test
    print(f"Train data: {x_train.shape} {y_train.shape}")
    print(f"Test data: {x_test.shape} {y_test.shape}")
    print()

    # Create network, then train and eval
    network_choice = Networks(args.network)
    networks = {Networks.LENET: LeNet, Networks.SIMPLE_LENET: SimpleLeNet, Networks.SUBFLOW: SubFlow}
    if network_choice == Networks.SUBFLOW:
        network = SubFlow(args.checkpoint_path, args.utilization, args.seed)
    else:
        network = networks[network_choice](args.checkpoint_path)
    print(network)
    print()

    # network.preload()
    network.train(x_train, y_train, args.epochs)
    network.evaluate(x_test, y_test)

    # Display some examples
    if args.display_examples:
        display_examples(network, test)


if __name__ == "__main__":
    main()
