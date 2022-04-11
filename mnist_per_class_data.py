from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

_CLASS_ID = 0
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class MNISTPerClass():
    def __init__(self, mnist, class_id) -> None:
        self._class_id = class_id
        self.train_images, self.train_labels = self.create_binary_dataset(
            mnist.train.images, mnist.train.labels, class_id, extend_samples=True)
        self.validation_images, self.validation_labels = self.create_binary_dataset(
            mnist.validation.images, mnist.validation.labels, class_id)
        self.test_images, self.test_labels = self.create_binary_dataset(
            mnist.test.images, mnist.test.labels, class_id)

    @staticmethod
    def create_binary_dataset(images, labels, class_id, extend_samples=False):
        pos_images, neg_images = [], []
        pos_labels, neg_labels = [], []
        for image, label in zip(images, labels):
            if np.argmax(label) == class_id:
                pos_images.append(image)
                pos_labels.append([1, 0])
            else:
                neg_images.append(image)
                neg_labels.append([0, 1])

        if extend_samples:
            remaining_samples = len(neg_images) - len(pos_images)
            assert remaining_samples >= 0

            while remaining_samples > 0:
                pos_images.extend(pos_images[:remaining_samples])
                pos_labels.extend(pos_labels[:remaining_samples])
                remaining_samples = len(neg_images) - len(pos_images)
                print(remaining_samples)
            assert remaining_samples == 0

        _images = pos_images + neg_images
        _labels = pos_labels + neg_labels
        return np.array(_images), np.array(_labels)


mnist_per_class = MNISTPerClass(mnist, class_id=_CLASS_ID)
print(
    f"MNIST per class train shape: {mnist_per_class.train_images.shape}, {mnist_per_class.train_labels.shape}")
print(
    f"MNIST per class val shape: {mnist_per_class.validation_images.shape}, {mnist_per_class.validation_labels.shape}")
print(
    f"MNIST per class test shape: {mnist_per_class.test_images.shape}, {mnist_per_class.test_labels.shape}")


def train_set():
    return mnist_per_class.train_images, mnist_per_class.train_labels


def validation_set():
    return mnist_per_class.validation_images, mnist_per_class.validation_labels


def test_set():
    return mnist_per_class.test_images, mnist_per_class.test_labels


def main():
    print('this is main')


if __name__ == '__main__':
    main()
