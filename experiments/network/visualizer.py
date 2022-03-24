"""
The Visualizer class provides (static) methods to visualize a network and its weight and bias in image form.
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colorbar import Colorbar
from matplotlib.gridspec import GridSpec
from typing import List, Optional

from .network import Network


class Visualizer:
    """
    The Visualizer class visualizes networks as images.
    """

    @staticmethod
    def network_to_image(network: Network) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def array_to_image(array: np.ndarray) -> np.ndarray:
        side_length = int(np.ceil(np.sqrt(array.size)))
        image = array.flatten()
        padding = (side_length ** 2) - image.size
        image = np.append(image, np.zeros(padding, dtype=np.float32))
        image = np.reshape(image, (side_length, side_length))
        assert image is not array
        return image

    @staticmethod
    def weights_to_image(weights: np.ndarray) -> np.ndarray:
        return Visualizer.array_to_image(weights)

    @staticmethod
    def bias_to_image(bias: np.ndarray) -> np.ndarray:
        return Visualizer.array_to_image(bias)

    @staticmethod
    def plot_network(network: Network,
                     output_file: str,
                     weight_vmin: Optional[float] = None,
                     weight_vmax: Optional[float] = None,
                     bias_vmin: Optional[float] = None,
                     bias_vmax: Optional[float] = None,
                     font_size: int = 4,
                     dpi: int = 300,
                     title_padding: int = 0,
                     show_figure: bool = True) -> None:
        """
        Plots the network using MATLAB plot and saves it to image file.

        :param network: The network.
        :param output_file: The filename for the output image.
        :param weight_vmin: Weight minimum value or None.
        :param weight_vmax: Weight maximum value or None.
        :param bias_vmin: Bias minimum value or None.
        :param bias_vmax: Bias maximum value or None.
        :param font_size: Font size.
        :param dpi: Figure DPI.
        :param title_padding: Title padding.
        :param show_figure: Flag whether to also show the image figure.
        :return:
        """

        figure = plt.figure(figsize=(network.count, 2), constrained_layout=True, dpi=dpi)
        figure.canvas.manager.set_window_title(f"{network.name} ({network.utilization:.2f} utilization)")
        figure.suptitle("Weights & Biases")
        grid = GridSpec(2, network.count + 1, figure, width_ratios=[3] * network.count + [1], height_ratios=[1, 1])

        last_weight_image = None
        last_bias_image = None
        for i, layer in enumerate(network.layers):
            weight_image = Visualizer.weights_to_image(layer.weight)
            bias_image = Visualizer.bias_to_image(layer.bias)

            weight_subplot = figure.add_subplot(grid[0, i])
            weight_subplot.set_xticks([])
            weight_subplot.set_yticks([])
            weight_subplot.set_title(f"{layer.weight_name} ({layer.weight.size})", size=font_size, pad=title_padding)
            last_weight_image = weight_subplot.imshow(weight_image, vmin=weight_vmin, vmax=weight_vmax)

            bias_subplot = figure.add_subplot(grid[1, i])
            bias_subplot.set_xticks([])
            bias_subplot.set_yticks([])
            bias_subplot.set_title(f"{layer.bias_name} ({layer.bias.size})", size=font_size, pad=title_padding)
            last_bias_image = bias_subplot.imshow(bias_image, vmin=bias_vmin, vmax=bias_vmax)

        weight_colorbar_axis = figure.add_subplot(grid[0, network.count])
        weight_colorbar_axis.tick_params(labelsize=font_size)
        Colorbar(mappable=last_weight_image, ax=weight_colorbar_axis)

        bias_colorbar_axis = figure.add_subplot(grid[1, network.count])
        bias_colorbar_axis.tick_params(labelsize=font_size)
        Colorbar(mappable=last_bias_image, ax=bias_colorbar_axis)

        plt.savefig(output_file, bbox_inches="tight", dpi=dpi)
        if show_figure:
            plt.show()

    @staticmethod
    def plot_arrays(arrays: List[np.ndarray],
                    output_file: str,
                    same_range: bool,
                    font_size: int = 4,
                    dpi: int = 300,
                    title_padding: int = 0,
                    show_figure: bool = True) -> None:
        """
        Plots the the given numpy arrays in a row using MATLAB plot and saves it to image file.

        :param arrays: A list of numpy arrays.
        :param output_file: The filename for the output image.
        :param same_range: Flag whether to use the maximum range over all arrays for each individual plot.
        :param font_size: Font size.
        :param dpi: Figure DPI.
        :param title_padding: Title padding.
        :param show_figure: Flag whether to also show the image figure.
        :return:
        """

        count = len(arrays)
        figure = plt.figure(figsize=(count, 1), constrained_layout=True, dpi=dpi)
        grid = GridSpec(1, count + 1, figure, width_ratios=[3] * count + [1])

        min_value = None
        max_value = None
        if same_range:
            all_data = np.concatenate([array.flatten() for array in arrays])
            min_value, max_value = np.min(all_data), np.max(all_data)

        last_image = None
        for i, array in enumerate(arrays):
            image = Visualizer.array_to_image(array)
            subplot = figure.add_subplot(grid[i])
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title(f"{i}", size=font_size, pad=title_padding)
            last_image = subplot.imshow(image, vmin=min_value, vmax=max_value)

        colorbar_axis = figure.add_subplot(grid[count])
        colorbar_axis.tick_params(labelsize=font_size)
        Colorbar(mappable=last_image, ax=colorbar_axis)

        plt.savefig(output_file, bbox_inches="tight", dpi=dpi)
        if show_figure:
            plt.show()
