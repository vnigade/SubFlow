"""
Evaluates the LeNet baseline model.
"""
import argparse
import os

from network import LeNet, LeNetConfiguration, SimpleLeNet, SimpleLeNetConfiguration
from network.mnist import load_data, display_examples


def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--model_base_directory", type=str, default="./models", help="The network model base directory.")
    args.add_argument("--epochs", type=int, default=5, help="The number of training epochs.")
    args.add_argument("--display_examples", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Displays some examples using MATPLOTLIB.")
    args = args.parse_args()

    # Load MNIST dataset
    _, test = load_data()
    x_test, y_test = test
    print(f"Test data: {x_test.shape} {y_test.shape}")
    print()

    # Create configurations to evaluate
    lenet_base_folder = os.path.join(args.model_base_directory, "LeNet")
    simple_lenet_base_folder = os.path.join(args.model_base_directory, "SimpleLeNet")
    configurations = [LeNetConfiguration(lenet_base_folder, args.epochs, leaky_relu=True),
                      LeNetConfiguration(lenet_base_folder, args.epochs, leaky_relu=False),
                      SimpleLeNetConfiguration(simple_lenet_base_folder, args.epochs, leaky_relu=True),
                      SimpleLeNetConfiguration(simple_lenet_base_folder, args.epochs, leaky_relu=False)]

    # Evaluate configurations
    for configuration in configurations:
        # Instantiate the network
        if isinstance(configuration, SimpleLeNetConfiguration):
            network = SimpleLeNet(initialization_directory=None, leaky_relu=configuration.leaky_relu)
        else:
            network = LeNet(initialization_directory=None, leaky_relu=configuration.leaky_relu)

        # Load the model
        model_directory = os.path.join(configuration.model_base_directory, configuration.path)
        network.load(model_directory)

        # Evaluate the network
        metrics = network.evaluate(x_test, y_test)
        print(f"{network.name} (epochs={configuration.epochs}, leaky_relu={configuration.leaky_relu}): {metrics}")

        # Display some examples
        if args.display_examples:
            display_examples(network, test)


if __name__ == "__main__":
    main()
