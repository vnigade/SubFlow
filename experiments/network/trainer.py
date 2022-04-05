"""
Defines training configurations and a Trainer class which trains a network for a configuration.
"""
import logging
import numpy as np
import os
import pathlib
import shutil
import sys

from dataclasses import dataclass
from typing import Optional, Union

from .networks import LeNet, SimpleLeNet, SubFlow


# =====================================================================================================================================================================================================
# Configurations
# =====================================================================================================================================================================================================

@dataclass
class BaseConfiguration:
    model_base_directory: str


@dataclass
class LeNetConfiguration(BaseConfiguration):
    epochs: int
    leaky_relu: bool = True

    @property
    def path(self):
        architecture_name = "leaky" if self.leaky_relu else "relu"
        return f"{architecture_name}/epochs{self.epochs}"


@dataclass
class SimpleLeNetConfiguration(BaseConfiguration):
    epochs: int
    leaky_relu: bool = True

    @property
    def path(self):
        architecture_name = "leaky" if self.leaky_relu else "relu"
        return f"{architecture_name}/epochs{self.epochs}"


@dataclass
class SubFlowLeNetConfiguration(BaseConfiguration):
    epochs: int
    leaky_relu: bool = True
    utilization: int = 100
    seed: int = 123456789
    initialization_directory: Optional[str] = None
    run: Optional[int] = None

    @property
    def path(self):
        architecture_name = "leaky" if self.leaky_relu else "relu"
        run = f"run{self.run}" if self.run else ""
        return f"{architecture_name}_{self.utilization}/epochs{self.epochs}/{run}"


Configuration = Union[LeNetConfiguration, SimpleLeNetConfiguration, SubFlowLeNetConfiguration]


# =====================================================================================================================================================================================================
# Trainer
# =====================================================================================================================================================================================================

class Trainer:
    """
    The Trainer class.
    """

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def __init__(self):
        self._configurations: list[Configuration] = list()

    def add_configuration(self, configuration: Configuration) -> None:
        self._configurations.append(configuration)

    def train(self, x: np.ndarray, y: np.ndarray, clear_contents: bool = True, verbose: bool = False) -> None:
        """
        Trains all the stored configurations.

        :param x: The input training data (features).
        :param y: The output training data (labels).
        :param clear_contents: Define whether the model directories should be fully cleared before training.
        :param verbose: Display logging information for each configuration in the terminal.
        :return: None.
        """

        stream_handler = None
        if verbose:
            stream_handler = logging.StreamHandler(sys.stdout)
            logging.getLogger().addHandler(stream_handler)

        for configuration in self._configurations:
            self._train_configuration(configuration, x, y, clear_contents)

        if stream_handler:
            logging.getLogger().removeHandler(stream_handler)

    # =================================================================================================================================================================================================
    # Private methods
    # =================================================================================================================================================================================================

    @staticmethod
    def _train_configuration(configuration: Configuration, x: np.ndarray, y: np.ndarray, clear_contents: bool) -> None:
        # Get the full model directory
        model_directory = os.path.join(configuration.model_base_directory, configuration.path)

        # Make sure that the directory exists and (if requested) clear its previous content
        if clear_contents and os.path.exists(model_directory):
            shutil.rmtree(model_directory)
        pathlib.Path(model_directory).mkdir(parents=True, exist_ok=True)

        # Setup logging to file in the model directory
        log_file = os.path.join(model_directory, "logging.log")
        log_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        log_handler = logging.FileHandler(filename=log_file, mode="w")
        log_handler.setFormatter(log_formatter)
        logger = logging.getLogger(model_directory)
        logger.addHandler(log_handler)
        logger.setLevel(logging.DEBUG)

        # Log configuration and training data information
        logger.info(f"{configuration}\n")
        logger.info(f"Train data: {x.shape} {y.shape}\n")

        # Create the network
        if isinstance(configuration, LeNetConfiguration):
            network = LeNet(initialization_directory=None, leaky_relu=configuration.leaky_relu)
        elif isinstance(configuration, SimpleLeNetConfiguration):
            network = SimpleLeNet(initialization_directory=None, leaky_relu=configuration.leaky_relu)
        elif isinstance(configuration, SubFlowLeNetConfiguration):
            network = SubFlow(configuration.initialization_directory, configuration.leaky_relu, configuration.utilization, configuration.seed)
        else:
            raise RuntimeError("Trainer tried to train an unknown network configuration.")
        logger.info(f"\n{network}\n")

        # Train the network
        network.train(model_directory, x, y, configuration.epochs)
