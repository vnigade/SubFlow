from __future__ import print_function
import numpy as np

import tensorflow.keras as keras
cifar10_train, cifar10_test = keras.datasets.cifar10.load_data()

print(f"Train shape: {cifar10_train[0].shape}, {cifar10_train[1].shape}")
print(f"Test shape: {cifar10_test[0].shape}, {cifar10_test[1].shape}")


def train_set():
    return cifar10_train[0], cifar10_train[1]


def validation_set():
    raise NotImplementedError


def test_set():
    return cifar10_test[0], cifar10_test[1]


def main():
    print('this is main')


if __name__ == '__main__':
    main()
