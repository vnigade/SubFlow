from __future__ import print_function
import numpy as np
from mnist_data import get_class_dataset
import tensorflow.keras as keras
cifar10_train, cifar10_test = keras.datasets.cifar10.load_data()

NUM_CLASSES = 10


def _preprocess(dataset):
    images, labels = dataset
    one_hot_labels = np.zeros(
        shape=(labels.shape[0], NUM_CLASSES), dtype=np.float)
    for i in range(labels.shape[0]):
        y = labels[i][0]
        one_hot_labels[i, y] = 1.0
    images = images / 255.0
    return (images, one_hot_labels)


def _get_validation_split(train_set, split_percentage=10):
    # TODO: Move this to a separate dataset_utils file
    n_images = train_set[0].shape[0]
    n_val_images = (n_images * split_percentage) // 100
    train_images, train_labels = (train_set[0][:-n_val_images],
                                  train_set[1][:-n_val_images])
    val_images, val_labels = (train_set[0][n_images-n_val_images:],
                              train_set[1][n_images-n_val_images:])
    return (train_images, train_labels), (val_images, val_labels)


train_set = _preprocess(cifar10_train)
cifar10_train, cifar10_val = _get_validation_split(
    train_set, split_percentage=2)
cifar10_test = _preprocess(cifar10_test)

print(
    f"Train shape: {cifar10_train[0].shape}, {cifar10_train[1].shape}")
print(f"Val shape: {cifar10_val[0].shape}, {cifar10_val[1].shape}")
print(f"Test shape: {cifar10_test[0].shape}, {cifar10_test[1].shape}")


def train_set(class_id=None):
    if class_id == None:
        return cifar10_train[0], cifar10_train[1]

    return get_class_dataset(cifar10_train[0], cifar10_train[1], class_id)


def validation_set(class_id=None):
    if class_id == None:
        return cifar10_val[0], cifar10_val[1]

    return get_class_dataset(cifar10_val[0], cifar10_val[1], class_id)


def test_set(class_id=None):
    if class_id == None:
        return cifar10_test[0], cifar10_test[1]

    return get_class_dataset(cifar10_test[0], cifar10_test[1], class_id)


def main():
    print('this is main')


if __name__ == '__main__':
    main()
