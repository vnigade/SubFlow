"""
Evaluates the LeNet baseline model.
"""
import argparse
import os

from network import LeNet, SimpleLeNet
from network.mnist import load_data, display_examples


def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--model_base_directory", type=str, default="./models", help="The network model base directory.")
    args.add_argument("--use_simple", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use the SimpleLeNet architecture instead of the default one.")
    args.add_argument("--leaky_relu", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use leaky ReLus instead of normal ones.")
    args.add_argument("--display_examples", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Displays some examples using MATPLOTLIB.")
    args = args.parse_args()

    # Load MNIST dataset
    _, test = load_data()
    x_test, y_test = test
    print(f"Test data: {x_test.shape} {y_test.shape}")
    print()

    # Create the network
    if args.use_simple:
        network = SimpleLeNet(initialization_directory=None, leaky_relu=args.leaky_relu)
    else:
        network = LeNet(initialization_directory=None, leaky_relu=args.leaky_relu)

    model_directory = os.path.join(args.model_base_directory, network.name + ("" if args.leaky_relu else "_relu"))
    print(f"Loading model: {model_directory}")
    network.load(model_directory)
    print(network)
    print()

    # Evaluate the network
    network.evaluate(x_test, y_test)

    # Display some examples
    if args.display_examples:
        display_examples(network, test)


if __name__ == "__main__":
    main()
