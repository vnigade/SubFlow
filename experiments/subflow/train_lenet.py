"""
Trains the LeNet baseline model.
"""
import argparse
import os

from network import LeNet, SimpleLeNet
from network.mnist import load_data


def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--model_base_directory", type=str, default="./models", help="The network model base directory.")
    args.add_argument("--epochs", type=int, default=5, help="The number of training epochs.")
    args.add_argument("--use_simple", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use the SimpleLeNet architecture instead of the default one.")
    args.add_argument("--leaky_relu", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use leaky ReLus instead of normal ones.")
    args = args.parse_args()

    # Load MNIST dataset
    train, test = load_data()
    (x_train, y_train), (x_test, y_test) = train, test
    print(f"Train data: {x_train.shape} {y_train.shape}")
    print(f"Test data: {x_test.shape} {y_test.shape}")
    print()

    # Create the network
    if args.use_simple:
        network = SimpleLeNet(initialization_directory=None, leaky_relu=args.leaky_relu)
    else:
        network = LeNet(initialization_directory=None, leaky_relu=args.leaky_relu)
    print(network)
    print()

    # Train the network
    output_directory = os.path.join(args.model_base_directory, network.name + ("" if args.leaky_relu else "_relu"))
    network.train(output_directory, x_train, y_train, args.epochs, clear_folder=True)
    network.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()
