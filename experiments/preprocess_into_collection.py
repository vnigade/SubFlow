"""
Collects the Network data extracted by the extract_from_subflow.py script and stores it into a NetworkCollection.

todo: currently this is hard-coded for MNIST dataset and 10 utilizations.

"""
import argparse
import os

from network import Network, NetworkCollection


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--base_data_folder", type=str, default="../output/", help="The base data folder.")
    args.add_argument("--output_filename", type=str, default="collection.pickle", help="The output file name.")
    args = args.parse_args()

    # Define input folders and utilization
    # todo: this should be passed in as arguments.
    inputs = {1.0: "network1",
              0.9: "sub_network9",
              0.8: "sub_network8",
              0.7: "sub_network7",
              0.6: "sub_network6",
              0.5: "sub_network5",
              0.4: "sub_network4",
              0.3: "sub_network3",
              0.2: "sub_network2",
              0.1: "sub_network1"}

    # Load networks
    networks = [Network.load_from_folder(args.base_data_folder, name, utilization) for utilization, name in inputs.items()]
    collection = NetworkCollection(networks)
    print(collection)

    # Save collection
    output_filename = os.path.join(args.base_data_folder, args.output_filename)
    collection.save(output_filename)


if __name__ == "__main__":
    main()
