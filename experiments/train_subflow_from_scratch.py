"""
Trains the SubFlow models.
"""
import argparse
import os

from network import SubFlowLeNetConfiguration, Trainer
from network.mnist import load_data


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

    # Create training configurations for training SubFlow from scratch
    subflow_base_folder = os.path.join(args.model_base_directory, "SubFlow/FromScratch/")
    trainer = Trainer()
    for utilization in range(10, 100, 10):
        configuration = SubFlowLeNetConfiguration(subflow_base_folder, args.epochs, leaky_relu=args.leaky_relu, utilization=utilization, seed=args.seed)
        trainer.add_configuration(configuration)

    # Train all the configurations
    trainer.train(x_train, y_train, clear_contents=args.clear_contents, verbose=True)


if __name__ == "__main__":
    main()
