"""
Evaluates the SubFlow model.
"""
import argparse
import os

from network import SubFlow, SubFlowLeNetConfiguration
from network.mnist import load_data, display_examples


def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--model_base_directory", type=str, default="./models", help="The network model base directory.")
    args.add_argument("--epochs", type=int, default=5, help="The number of training epochs.")
    args.add_argument("--seed", type=int, default=123456789, help="The random seed for activation mask sampling.")
    args.add_argument("--display_examples", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Displays some examples using MATPLOTLIB.")
    args = args.parse_args()

    # Load MNIST dataset
    _, test = load_data()
    x_test, y_test = test
    print(f"Test data: {x_test.shape} {y_test.shape}")
    print()

    # Create configurations to evaluate
    subflow_base_folder = os.path.join(args.model_base_directory, "SubFlow/FromScratch/")
    configurations = list()
    for utilization in range(10, 100, 10):
        configurations.append(SubFlowLeNetConfiguration(subflow_base_folder, args.epochs, leaky_relu=True, utilization=utilization, seed=args.seed))

    # Evaluate configurations
    for configuration in configurations:
        # Instantiate and load the model
        network = SubFlow(None, configuration.leaky_relu, configuration.utilization, configuration.seed)
        model_directory = os.path.join(configuration.model_base_directory, configuration.path)
        network.load(model_directory)

        # Evaluate the network
        metrics = network.evaluate(x_test, y_test)
        print(f"{network.name} (epochs={configuration.epochs}, leaky_relu={configuration.leaky_relu}, utilization={configuration.utilization}): {metrics}")

        # Display some examples
        if args.display_examples:
            display_examples(network, test)


if __name__ == "__main__":
    main()
