"""
Visualizes weights and biases that were extracted using the extract_data.py script.
"""
import argparse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =================================================================================================
# Network Loading
# =================================================================================================

@dataclass
class Network:
    name: str
    named_weights: Dict[str, np.ndarray]
    named_biases: Dict[str, np.ndarray]

    def __post_init__(self):
        assert len(self.named_weights) == len(self.named_biases)
        for values in self.named_weights.values():
            assert values.dtype == np.float32
        for values in self.named_biases.values():
            assert values.dtype == np.float32

    def __repr__(self):
        return f"Network '{self.name}' [{self.count}]"

    @property
    def count(self) -> int:
        return len(self.named_weights)

    @property
    def weight_names(self) -> List[str]:
        return list(self.named_weights.keys())

    @property
    def bias_names(self) -> List[str]:
        return list(self.named_biases.keys())

    @property
    def weights(self) -> List[np.ndarray]:
        return list(self.named_weights.values())

    @property
    def biases(self) -> List[np.ndarray]:
        return list(self.named_biases.values())

    @property
    def minimum_weight_size(self) -> int:
        return np.min([w.size for w in self.weights])

    @property
    def maximum_weight_size(self) -> int:
        return np.max([w.size for w in self.weights])

    @property
    def minmax_weight_value(self) -> Tuple[float, float]:
        stacked_weights = np.concatenate([w.flatten() for w in self.weights])
        return np.min(stacked_weights), np.max(stacked_weights)

    @property
    def minimum_bias_size(self) -> int:
        return np.min([b.size for b in self.biases])

    @property
    def maximum_bias_size(self) -> int:
        return np.max([b.size for b in self.biases])

    @property
    def minmax_bias_value(self) -> Tuple[float, float]:
        stacked_biases = np.concatenate([b.flatten() for b in self.biases])
        return np.min(stacked_biases), np.max(stacked_biases)


def load_network_data(base_data_folder: str, network_name: str) -> Network:
    data_folder = os.path.join(base_data_folder, network_name)
    files = os.listdir(data_folder)
    weight_files = [f for f in files if f.startswith("weight_")]
    bias_files = [f for f in files if f.startswith("bias_")]
    assert len(weight_files) == len(bias_files)

    weights = {os.path.splitext(f)[0]: np.load(os.path.join(data_folder, f)) for f in weight_files}
    biases = {os.path.splitext(f)[0]: np.load(os.path.join(data_folder, f)) for f in bias_files}
    return Network(network_name, weights, biases)


# =================================================================================================
# Visualization
# =================================================================================================

def weights_to_image(weights: np.ndarray) -> np.ndarray:
    side_length = int(np.ceil(np.sqrt(weights.size)))
    image = weights.flatten()
    padding = (side_length ** 2) - image.size
    image = np.append(image, np.zeros(padding, dtype=np.float32))
    image = np.reshape(image, (side_length, side_length))
    assert image is not weights
    return image


def bias_to_image(bias: np.ndarray) -> np.ndarray:
    return weights_to_image(bias)


def plot_network(network: Network, weight_vmin: Optional[float] = None, weight_vmax: Optional[float] = None, bias_vmin: Optional[float] = None, bias_vmax: Optional[float] = None,
                 show_figure: bool = True) -> None:
    # figure, axes = plt.subplots(2, network.count)
    # figure = plt.figure(figsize=(2, network.count))
    figure = plt.figure()
    figure.canvas.manager.set_window_title(network.name)
    figure.suptitle("Weights & Biases")

    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)

    gs = gridspec.GridSpec(2, network.count)
    gs.update(wspace=0.15, hspace=0.05)

    for i in range(network.count):
        weight_name = f"weight_{i}"
        weight = network.named_weights[weight_name]
        weight_image = weights_to_image(weight)

        weight_axis = plt.subplot(gs[0, i])
        weight_axis_image = weight_axis.imshow(weight_image, vmin=weight_vmin, vmax=weight_vmax)
        weight_axis.set_title(weight_name)
        figure.colorbar(weight_axis_image, ax=weight_axis, location="right")

        bias_name = f"bias_{i}"
        bias = network.named_biases[bias_name]
        bias_image = bias_to_image(bias)

        bias_axis = plt.subplot(gs[1, i])
        bias_axis_image = bias_axis.imshow(bias_image, vmin=bias_vmin, vmax=bias_vmax)
        bias_axis.set_title(bias_name)
        figure.colorbar(bias_axis_image, ax=bias_axis, location="right")

    # plt.tight_layout()
    plt.savefig(f"{network.name}.png", bbox_inches="tight", dpi=300)
    if show_figure:
        plt.show()


# =================================================================================================
# Main
# =================================================================================================

def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--base_data_folder", type=str, default="../output/", help="The base data folder.")
    args.add_argument("--networks", "--named-list", type=str, default=["network1"] + [f"sub_network{i}" for i in range(1, 10)], help="The name of the network.")
    # args.add_argument("--networks", "--named-list", type=str, default=["network1"], help="The name of the network.")
    args = args.parse_args()

    # Load networks
    networks = [load_network_data(args.base_data_folder, network) for network in args.networks]
    for network in networks:
        print(network)

    # Compute the min/max values for weights and biases
    minmax_weights = [n.minmax_weight_value for n in networks]
    min_weights, max_weights = zip(*minmax_weights)
    min_weight_value = np.min(min_weights)
    max_weight_value = np.max(max_weights)

    minmax_biases = [n.minmax_bias_value for n in networks]
    min_biases, max_biases = zip(*minmax_biases)
    min_bias_value = np.min(min_biases)
    max_bias_value = np.max(max_biases)

    # Plot networks
    for network in networks:
        plot_network(network, weight_vmin=min_weight_value, weight_vmax=max_weight_value, bias_vmin=min_bias_value, bias_vmax=max_bias_value, show_figure=False)


if __name__ == "__main__":
    main()
