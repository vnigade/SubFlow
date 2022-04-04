"""
Trains a standard Google LeNet on the MNIST dataset.
"""
import argparse
import enum
import os
import tensorflow as tf

from network import LeNet, SimpleLeNet, SubFlow
from network.mnist import load_data, display_examples


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
    args.add_argument("--model_base_directory", type=str, default="./models", help="The network model base directory.")
    args.add_argument("--network", type=str, default="subflow", choices=[n.value for n in Networks], help="The network to run.")
    args.add_argument("--epochs", type=int, default=5, help="The number of training epochs.")
    args.add_argument("--leaky_relu", default=True, help="Use leaky ReLus instead of normal ones.")
    args.add_argument("--utilization", type=int, default=20, help="The network utilization in percentage (integer values 1 to 100) [SubFlow only].")
    args.add_argument("--seed", type=int, default=123456789, help="The random seed for activation mask sampling [SubFlow only].")
    args.add_argument("--display_examples", default=False, help="If True, displays some examples using MATPLOTLIB.")
    args = args.parse_args()

    # Load MNIST dataset
    train, test = load_data()
    (x_train, y_train), (x_test, y_test) = train, test
    print(f"Train data: {x_train.shape} {y_train.shape}")
    print(f"Test data: {x_test.shape} {y_test.shape}")
    print()

    # Create network, then train and eval
    network_choice = Networks(args.network)
    if network_choice == Networks.LENET:
        network = LeNet(None, args.leaky_relu)
    elif network_choice == Networks.SIMPLE_LENET:
        network = SimpleLeNet(None, args.leaky_relu)
    elif network_choice == Networks.SUBFLOW:
        network = SubFlow(None, args.leaky_relu, args.utilization, args.seed)
    else:
        raise NotImplementedError(f"Unknown network choice: {network_choice}")
    print(network)
    print()

    # network.preload()
    output_directory = os.path.join(args.model_base_directory, network.name)
    network.train(output_directory, x_train, y_train, args.epochs, clear_folder=True)
    network.evaluate(x_test, y_test)

    # Display some examples
    if args.display_examples:
        display_examples(network, test)


if __name__ == "__main__":
    main()
