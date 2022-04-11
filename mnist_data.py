from __future__ import print_function
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

NUM_CLASSES = 10
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(f"Train shape: {mnist.train.images.shape}, {mnist.train.labels.shape}")
print(f"Test shape: {mnist.test.images.shape}, {mnist.test.labels.shape}")


def get_class_dataset(full_images, full_labels, class_id):
    # @TODO: move this to a seperate dataset_utils file.
    images, labels = [], []
    for image, label in zip(full_images, full_labels):
        if np.argmax(label) == class_id:
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


def train_set(class_id=None):
    """
    Return images and labels from the train set

    @param
        class_id    Returns dataset only for this class
    """
    if class_id == None:
        return mnist.train.images, mnist.train.labels

    return get_class_dataset(mnist.train.images, mnist.train.labels, class_id)


def validation_set(class_id=None):
    """
    Return images and labels from the validation set

    @param
        class_id    Returns dataset only for this class
    """
    if class_id == None:
        return mnist.validation.images, mnist.validation.labels

    return get_class_dataset(mnist.validation.images, mnist.validation.labels, class_id)


def test_set(class_id=None):
    """
    Return images and labels from the test set

    @param
        class_id    Returns dataset only for this class
    """
    if class_id == None:
        return mnist.test.images, mnist.test.labels

    return get_class_dataset(mnist.test.images, mnist.test.labels, class_id)


def main():
    print('this is main')


if __name__ == '__main__':
    main()
