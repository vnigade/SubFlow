"""
Evaluates the SubFlow model.
"""
import argparse
import os
import pathlib

# Disable tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from collections import defaultdict
from network import BaseConfiguration, SubFlow, SubFlowLeNetConfiguration
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

    # Specify which training strategies to evaluate
    training_strategies = ["FromScratch", "FromLeNet", "Progressive"]

    # Find all configurations for the training strategies
    subflow_base_folder = os.path.join(args.model_base_directory, "SubFlow")
    strategy_configurations = defaultdict(list)
    for strategy in training_strategies:
        strategy_folder = os.path.join(subflow_base_folder, strategy)
        for config_file in pathlib.Path(strategy_folder).rglob("train.configuration"):
            configuration = BaseConfiguration.load_from_file(str(config_file))
            assert isinstance(configuration, SubFlowLeNetConfiguration)
            # Group configurations by training strategy and number of training epochs
            strategy_configurations[(strategy, configuration.epochs)].append(configuration)

    # Evaluate strategies
    for (strategy, epochs), configurations in strategy_configurations.items():
        # Display strategy header
        print("*" * 100)
        print(f"Training strategy: {strategy} - Epochs: {epochs}")
        print("*" * 100)

        # Evaluate configurations for strategy
        for configuration in configurations:
            # Instantiate and load the model
            network = SubFlow(None, configuration.leaky_relu, configuration.utilization, configuration.seed)
            model_directory = os.path.join(configuration.model_base_directory, configuration.path)
            network.load(model_directory)

            # Evaluate the network
            metrics = network.evaluate(x_test, y_test)

            # Print metrics
            print(f"utilization={configuration.utilization}, leaky_relu={configuration.leaky_relu}, initialization_directory={configuration.initialization_directory}: {metrics}")

            # Display some examples
            if args.display_examples:
                display_examples(network, test)

        print()


if __name__ == "__main__":
    main()
