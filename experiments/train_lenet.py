"""
Trains the LeNet baseline models.
"""
import argparse
import os

from data.mnist import load_data
from network import LeNetConfiguration, SimpleLeNetConfiguration, Trainer


def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--model_base_directory", type=str, default="./models", help="The network model base directory.")
    args.add_argument("--epochs", type=int, default=5, help="The number of training epochs.")
    args.add_argument("--clear_contents", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Clears the previous contents of the output directory if set.")
    args = args.parse_args()

    # Load MNIST dataset
    train, _ = load_data()
    x_train, y_train = train

    # Create training configurations for multiple LeNet architectures
    lenet_base_folder = os.path.join(args.model_base_directory, "LeNet")
    simple_lenet_base_folder = os.path.join(args.model_base_directory, "SimpleLeNet")
    trainer = Trainer()
    trainer.add_configuration(LeNetConfiguration(lenet_base_folder, args.epochs, leaky_relu=True))
    trainer.add_configuration(LeNetConfiguration(lenet_base_folder, args.epochs, leaky_relu=False))
    trainer.add_configuration(SimpleLeNetConfiguration(simple_lenet_base_folder, args.epochs, leaky_relu=True))
    trainer.add_configuration(SimpleLeNetConfiguration(simple_lenet_base_folder, args.epochs, leaky_relu=False))

    # Train all the configurations
    trainer.train(x_train, y_train, clear_contents=args.clear_contents, verbose=True)


if __name__ == "__main__":
    main()
