from __future__ import print_function
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(f"Train shape: {mnist.train.images.shape}, {mnist.train.labels.shape}")
print(f"Test shape: {mnist.test.images.shape}, {mnist.test.labels.shape}")

def train_set():
    return mnist.train.images, mnist.train.labels

def validation_set():
    return mnist.validation.images, mnist.validation.labels

def test_set():
    return mnist.test.images, mnist.test.labels

def main():
    print('this is main')

if __name__ == '__main__':
    main()
