"""
Trains the LeNet baseline model.
"""
import argparse
import logging
import os
import shutil
import pathlib
import sys

from network import LeNet, SimpleLeNet
from network.mnist import load_data


def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--model_base_directory", type=str, default="./models", help="The network model base directory.")
    args.add_argument("--epochs", type=int, default=5, help="The number of training epochs.")
    args.add_argument("--use_simple", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use the SimpleLeNet architecture instead of the default one.")
    args.add_argument("--leaky_relu", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use leaky ReLus instead of normal ones.")
    args.add_argument("--clear_folder", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Clears the previous contents of the output directory if set.")
    args = args.parse_args()

    # Define output directory
    network_name = "SimpleLeNet" if args.use_simple else "LeNet"
    if not args.leaky_relu:
        network_name += "_relu"
    output_directory = os.path.join(args.model_base_directory, network_name)

    # Make sure that the directory exists and (if requested) clear its previous content
    if args.clear_folder and os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_directory, "logging.log")
    logging.basicConfig(filename=log_file, filemode="w", format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    # Load MNIST dataset
    train, _ = load_data()
    x_train, y_train = train
    logging.info(f"Train data: {x_train.shape} {y_train.shape}\n")

    # Create the network
    if args.use_simple:
        network = SimpleLeNet(initialization_directory=None, leaky_relu=args.leaky_relu)
    else:
        network = LeNet(initialization_directory=None, leaky_relu=args.leaky_relu)
    logging.info(f"\n{network}\n")

    # Train the network
    network.train(output_directory, x_train, y_train, args.epochs)


if __name__ == "__main__":
    main()
