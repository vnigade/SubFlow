"""
Trains the SubFlow models from scratch.
"""
import argparse
import os

from data.mnist import load_data
from network import LeNetConfiguration, SubFlowLeNetConfiguration, Trainer


# =================================================================================================
# Training strategies
# =================================================================================================

def create_from_scratch_configurations(base_folder: str, epochs: int, leaky_relu: bool, seed: int) -> list[SubFlowLeNetConfiguration]:
    configurations = list()
    for utilization in range(10, 100 + 1, 10):
        configuration = SubFlowLeNetConfiguration(base_folder, epochs, leaky_relu=leaky_relu, utilization=utilization, seed=seed)
        configurations.append(configuration)
    return configurations


def create_from_lenet_configurations(base_folder: str, epochs: int, leaky_relu: bool, seed: int, lenet_base_folder: str, lenet_epochs: int) -> list[SubFlowLeNetConfiguration]:
    # Get model path of the base LeNet model for initialization
    lenet_configuration = LeNetConfiguration(lenet_base_folder, lenet_epochs, leaky_relu)
    lenet_model_directory = os.path.join(lenet_configuration.model_base_directory, lenet_configuration.path)

    # Create training configurations for training SubFlow from the LeNet baseline
    configurations = list()
    for utilization in range(10, 100 + 1, 10):
        configuration = SubFlowLeNetConfiguration(base_folder, epochs, leaky_relu=leaky_relu, utilization=utilization, seed=seed, initialization_directory=lenet_model_directory)
        configurations.append(configuration)
    return configurations


def create_progressive_configurations(base_folder: str, epochs: int, leaky_relu: bool, seed: int, lenet_base_folder: str) -> list[SubFlowLeNetConfiguration]:
    # Get model path of the base LeNet model for initialization
    lenet_configuration = LeNetConfiguration(lenet_base_folder, epochs, leaky_relu)
    lenet_model_directory = os.path.join(lenet_configuration.model_base_directory, lenet_configuration.path)

    # Create training configurations for training SubFlow from the LeNet baseline
    previous_model = lenet_model_directory
    configurations = list()
    for utilization in reversed(range(10, 100 + 1, 10)):
        configuration = SubFlowLeNetConfiguration(base_folder, epochs, leaky_relu=leaky_relu, utilization=utilization, seed=seed, initialization_directory=previous_model)
        configurations.append(configuration)
        previous_model = os.path.join(base_folder, configuration.path)
    return configurations


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--model_base_directory", type=str, default="./models", help="The network model base directory.")
    args.add_argument("--epochs", type=int, default=5, help="The number of training epochs.")
    args.add_argument("--leaky_relu", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use leaky ReLus instead of normal ones.")
    args.add_argument("--seed", type=int, default=123456789, help="The random seed for activation mask sampling.")
    args.add_argument("--clear_contents", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Clears the previous contents of the output directory if set.")
    args = args.parse_args()

    # Load MNIST dataset
    train, _ = load_data()
    x_train, y_train = train

    # Create training configurations
    lenet_base_folder = os.path.join(args.model_base_directory, "LeNet")
    subflow_base_folder = os.path.join(args.model_base_directory, "SubFlow")
    configurations = create_from_lenet_configurations(os.path.join(subflow_base_folder, "FromLeNet"), 0, args.leaky_relu, args.seed, lenet_base_folder, args.epochs)
    configurations += create_from_lenet_configurations(os.path.join(subflow_base_folder, "FromLeNet"), args.epochs, args.leaky_relu, args.seed, lenet_base_folder, args.epochs)
    configurations += create_from_scratch_configurations(os.path.join(subflow_base_folder, "FromScratch"), args.epochs, args.leaky_relu, args.seed)
    configurations += create_progressive_configurations(os.path.join(subflow_base_folder, "Progressive"), args.epochs, args.leaky_relu, args.seed, lenet_base_folder)

    # Create a trainer and train all the configurations
    trainer = Trainer()
    for configuration in configurations:
        trainer.add_configuration(configuration)
    trainer.train(x_train, y_train, clear_contents=args.clear_contents, verbose=True)


if __name__ == "__main__":
    main()
