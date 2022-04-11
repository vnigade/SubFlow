from __future__ import print_function
from tensorflow.python.framework import ops
from tensorflow.python.client import timeline
import numpy as np
import tensorflow as tf
import os
import copy
import pickle
import sys
import argparse
import shutil
import time
import models


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
save_filename = os.path.splitext(os.path.basename(__file__))[0] + '.obj'

sub_conv2d_library = tf.load_op_library(
    './sub_conv2d.so')
sub_matmul_library = tf.load_op_library(
    './sub_matmul.so')


@ops.RegisterGradient("SubConv")
def _sub_conv_grad(op, grad_ys):
    grad_list = []
    for i in range(len(op.inputs)):
        grad = None
        if i == 0:
            input = op.inputs[0]
            filter = op.inputs[1]
            stride = op.inputs[2]
            what_to = op.inputs[3]
            where_to = op.inputs[4]
            conv_len = op.inputs[5]
            activation = op.inputs[6]
            grad = sub_conv2d_library.sub_conv_back_input(input, filter, stride,
                                                          what_to, where_to, conv_len, activation, grad_ys)
        elif i == 1:
            input = op.inputs[0]
            filter = op.inputs[1]
            stride = op.inputs[2]
            what_to = op.inputs[3]
            where_to = op.inputs[4]
            conv_len = op.inputs[5]
            activation = op.inputs[6]
            grad = sub_conv2d_library.sub_conv_back_filter(input, filter, stride,
                                                           what_to, where_to, conv_len, activation, grad_ys)
        else:
            grad = None

        grad_list.append(grad)

    return grad_list


@ops.RegisterGradient("SubMatmul")
def _sub_matmul_grad(op, grad_ys):
    grad_list = []
    for i in range(len(op.inputs)):
        grad = None
        if i == 0:
            grad = tf.matmul(grad_ys, tf.transpose(op.inputs[1]))
        elif i == 1:
            grad = sub_matmul_library.sub_matmul_back_mat_b(op.inputs[0],
                                                            op.inputs[1], op.inputs[2], grad_ys)
        else:
            grad = None

        grad_list.append(grad)

    return grad_list


class SubFlow:
    def __init__(self):
        self.network_no = 1
        self.network_list = []
        self.subflow_network_no = 1
        self.sub_network_list = []

    def constructNetwork(self, layers, name=None):
        if name is None:
            network_name = "Network_" + str(self.network_no)
        else:
            network_name = name

        network = Network(self.network_no, layers, network_name)
        self.network_list.append((self.network_no, network, network_name))
        self.network_no += 1

    def destructNetwork(self, network_no):
        print("destructNetwork %d" % network_no)
        for network in self.network_list:
            if network[0] == network_no:
                self.network_list.remove(network)
                if os.path.exists(network[1].network_dir):
                    shutil.rmtree(network[1].network_dir)
                self.network_no -= 1
                return

    def construct_sub_network(self, network, name=None):
        if name is None:
            network_name = "SubFlowNetwork_" + str(self.subflow_network_no)
        else:
            network_name = name

        network = SubFlowNetwork(
            self.subflow_network_no, network, network_name)
        self.sub_network_list.append(
            (self.subflow_network_no, network, network_name))
        self.subflow_network_no += 1

    def destruct_sub_network(self, subflow_network_no):
        print("destruct_sub_network %d" % subflow_network_no)
        for sub_network in self.sub_network_list:
            if sub_network[0] == subflow_network_no:
                self.sub_network_list.remove(sub_network)
                if os.path.exists(sub_network[1].network_dir):
                    shutil.rmtree(sub_network[1].network_dir)
                self.subflow_network_no -= 1
                return


class Network:
    def __init__(self, network_no, layers_str, network_name=None):
        self.network_no = network_no
        self.network_name = network_name
        self.layers = self.parse_layers(layers_str)
        self.layer_type, self.num_of_neuron_per_layer, self.num_of_neuron_per_layer_without_pool, self.num_of_weight_per_layer,\
            self.num_of_bias_per_layer = self.calculate_num_of_weight(
                self.layers)

        self.num_of_neuron = 0
        for layer in self.num_of_neuron_per_layer:
            self.num_of_neuron += np.prod(layer)

        self.num_of_weight = sum(self.num_of_weight_per_layer)
        self.num_of_bias = sum(self.num_of_bias_per_layer)

        self.network_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        'network' + str(self.network_no))

        self.network_file_name = 'network' + str(self.network_no)
        self.network_file_path = os.path.join(
            self.network_dir, self.network_file_name)

        self.importance_file_name = 'importance'
        self.importance_file_path = os.path.join(
            self.network_dir, self.importance_file_name + '.npy')

        self.parameter_file_name = 'parameter'
        self.parameter_file_path = os.path.join(
            self.network_dir, self.parameter_file_name + '.npy')

        self.neuron_base_name = "neuron_"
        self.weight_base_name = "weight_"
        self.bias_base_name = "bias_"
        self.output_base_name = "output_"

        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                self.buildNetwork(sess)
                if not os.path.exists(self.network_dir):
                    os.makedirs(self.network_dir)
                self.saveNetwork(sess)

    def buildNetwork(self, sess):
        layer_type = copy.deepcopy(self.layer_type)
        layer_type = list(filter(lambda type: type != 'max_pool', layer_type))
        layers = self.layers
        parameters = {}
        neurons = {}
        outputs = {}
        parameters_to_regularize = []

        keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        neurons[0] = tf.placeholder(tf.float32, [None]+layers[0],
                                    name=self.neuron_base_name+'0')
        outputs[0] = tf.identity(neurons[0], name=self.output_base_name+'0')
        print(neurons[0])

        for layer_no in range(1, len(layers)):
            weight_name = self.weight_base_name + str(layer_no-1)
            bias_name = self.bias_base_name + str(layer_no-1)
            neuron_name = self.neuron_base_name + str(layer_no)
            output_name = self.output_base_name + str(layer_no)

            if layer_type[layer_no] == "conv":
                conv_parameter = {
                    'weights': tf.get_variable(weight_name,
                                               shape=(layers[layer_no]),
                                               initializer=tf.contrib.layers.xavier_initializer()),
                    'biases': tf.get_variable(bias_name,
                                              shape=(layers[layer_no][3]),
                                              initializer=tf.contrib.layers.xavier_initializer()),
                }

                parameters_to_regularize.append(tf.reshape(conv_parameter['weights'],
                                                           [tf.size(conv_parameter['weights'])]))
                parameters_to_regularize.append(tf.reshape(conv_parameter['biases'],
                                                           [tf.size(conv_parameter['biases'])]))

                parameters[layer_no-1] = conv_parameter
                print('conv_parameter', parameters[layer_no-1])

                rank = sess.run(tf.rank(neurons[layer_no-1]))

                for _ in range(4 - rank):
                    neurons[layer_no -
                            1] = tf.expand_dims(neurons[layer_no-1], -1)

                # convolution
                strides = 1
                output = tf.nn.conv2d(neurons[layer_no-1],
                                      conv_parameter['weights'],
                                      strides=[1, strides, strides, 1], padding='VALID')
                output_biased = tf.nn.bias_add(
                    output, conv_parameter['biases'])

                # max pooling
                k = 2
                activation = tf.nn.leaky_relu(output_biased, name=output_name)
                outputs[layer_no] = activation

                if len(layers[layer_no+1]) < 4:
                    print("LAST CNN BEFORE FC")
                    neuron = tf.nn.max_pool(activation,
                                            ksize=[1, k, k, 1],
                                            strides=[1, k, k, 1], padding='VALID')
                    neuron = tf.transpose(
                        neuron, [0, 3, 1, 2], name=neuron_name)
                else:
                    neuron = tf.nn.max_pool(activation,
                                            ksize=[1, k, k, 1],
                                            strides=[1, k, k, 1], padding='VALID', name=neuron_name)

                neurons[layer_no] = neuron

            elif layer_type[layer_no] == "hidden" or layer_type[layer_no] == "output":
                fc_parameter = {
                    'weights': tf.get_variable(weight_name,
                                               shape=(np.prod(self.num_of_neuron_per_layer[layer_no-1]),
                                                      np.prod(self.num_of_neuron_per_layer[layer_no])),
                                               initializer=tf.contrib.layers.xavier_initializer()),
                    'biases': tf.get_variable(bias_name,
                                              shape=(
                                                  np.prod(self.num_of_neuron_per_layer[layer_no])),
                                              initializer=tf.contrib.layers.xavier_initializer()),
                }

                parameters_to_regularize.append(tf.reshape(fc_parameter['weights'],
                                                           [tf.size(fc_parameter['weights'])]))
                parameters_to_regularize.append(tf.reshape(fc_parameter['biases'],
                                                           [tf.size(fc_parameter['biases'])]))

                parameters[layer_no-1] = fc_parameter
                print('fc_parameter', parameters[layer_no-1])

                # fully-connected
                flattened = tf.reshape(neurons[layer_no-1],
                                       [-1, np.prod(self.num_of_neuron_per_layer[layer_no-1])])
                neuron_drop = tf.nn.dropout(flattened, rate=1 - keep_prob)

                if layer_type[layer_no] == "hidden":
                    neuron = tf.nn.leaky_relu(tf.add(tf.matmul(neuron_drop,
                                                               fc_parameter['weights']), fc_parameter['biases']),
                                              name=neuron_name)
                elif layer_type[layer_no] == "output":
                    y_b = tf.add(tf.matmul(neuron_drop, fc_parameter['weights']),
                                 fc_parameter['biases'])
                    neuron = tf.divide(tf.exp(y_b-tf.reduce_max(y_b)),
                                       tf.reduce_sum(
                                           tf.exp(y_b-tf.reduce_max(y_b))),
                                       name=neuron_name)

                neurons[layer_no] = neuron
                outputs[layer_no] = tf.identity(neuron, name=output_name)
            print(neuron)

        # input
        x = neurons[0]

        # output
        y = neurons[len(layers)-1]

        # correct labels
        y_ = tf.placeholder(tf.float32, [None] + layers[-1], name='y_')

        # define the loss function
        regularization = 0.00001 * \
            tf.nn.l2_loss(tf.concat(parameters_to_regularize, 0))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                                      reduction_indices=[1]), name='cross_entropy') + regularization

        # define accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1),
                                      name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                  name='accuracy')

        # for training
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           name='optimizer').minimize(cross_entropy)

        init = tf.global_variables_initializer()
        sess.run(init)

    def loadNetwork(self, sess):
        saver = tf.train.import_meta_graph(self.network_file_path + '.meta')
        saver.restore(sess, self.network_file_path)

    def saveNetwork(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.network_file_path)
        parameter = sess.run(tf.trainable_variables())
        np.save(self.parameter_file_path, parameter, allow_pickle=True)

    def doTrain(self, sess, graph, train_set, validation_set, batch_size,
                train_iteration, epochs):
        print("doTrain")

        x = graph.get_tensor_by_name("neuron_0:0")
        y = graph.get_tensor_by_name(f"neuron_{len(self.layers)-1}:0")
        y_ = graph.get_tensor_by_name("y_:0")
        keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        optimizer = graph.get_operation_by_name("optimizer")

        # Reshape the input data into NHWC. @NOTE: The layers are defined is in NHWC.
        input_data_validation_reshaped = np.reshape(validation_set[0],
                                                    ([-1] + x.get_shape().as_list()[1:]))
        print(
            f"doTrain, Input data shape {x.get_shape().as_list()}, reshaped data shape {input_data_validation_reshaped.shape}")
        labels_validation = validation_set[1]

        highest_accuracy = 0

        n_train_samples = train_set[0].shape[0]
        train_iteration = (n_train_samples // batch_size)
        print(f"doTrain: Total iterations {epochs * train_iteration} ")
        for epoch in range(epochs):
            indices = np.arange(n_train_samples)
            np.random.shuffle(indices)
            for i in range(train_iteration):
                input_data, labels = self.next_batch(
                    train_set, batch_size, i, indices)
                # Reshape the input data into NHWC. @NOTE: The layers are defined is in NHWC.
                input_data_reshaped = \
                    np.reshape(
                        input_data, ([-1] + x.get_shape().as_list()[1:]))

                sess.run(optimizer, feed_dict={x: input_data_reshaped,
                                               y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})

            val_accuracy = sess.run(accuracy, feed_dict={
                x: input_data_validation_reshaped, y_: labels_validation,
                keep_prob_input: 1.0, keep_prob: 1.0})
            print("epoch %d, Validation accuracy: %f" %
                  (epoch, val_accuracy))

            if val_accuracy > highest_accuracy:
                highest_accuracy = val_accuracy
                self.saveNetwork(sess)

    def train(self, train_set, validation_set, batch_size, train_iteration, epochs=10):
        print("train")
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                self.loadNetwork(sess)
                self.doTrain(sess, graph, train_set, validation_set, batch_size,
                             train_iteration, epochs)

    def doInfer(self, sess, graph, data_set, label=None):
        tensor_x_name = "neuron_0:0"
        x = graph.get_tensor_by_name(tensor_x_name)
        tensor_y_name = "neuron_" + str(len(self.layers)-1) + ":0"
        y = graph.get_tensor_by_name(tensor_y_name)
        keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        # infer
        data_set_reshaped = np.reshape(
            data_set, ([-1] + x.get_shape().as_list()[1:]))
        infer_result = sess.run(y, feed_dict={
            x: data_set_reshaped, keep_prob_input: 1.0, keep_prob: 1.0})

        test_accuracy = None
        if label is not None:
            predicted_samples = 0
            total_samples = label.shape[0]
            for pred, gt in zip(infer_result, label):
                if np.argmax(pred) == np.argmax(gt):
                    predicted_samples += 1
            test_accuracy = predicted_samples / total_samples
            print("Validation Inference accuracy:1.0=%f" % test_accuracy)

        return infer_result

    def infer(self, data_set, label=None):
        print("infer")
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                self.loadNetwork(sess)
                return self.doInfer(sess, graph, data_set, label)

    def next_batch(self, data_set, batch_size, batch_index, indices):
        data = data_set[0]
        label = data_set[1]  # one-hot vectors

        # XXX: So horrible.
        # The training iteration should include random items
        # in one interation but it should include all the inputs
        # data_num = np.random.choice(
        #     data.shape[0], size=batch_size, replace=False)
        start_index = batch_index*batch_size
        end_index = start_index + batch_size
        data_num = indices[start_index: end_index]

        batch = data[data_num, :]
        label = label[data_num, :]  # one-hot vectors

        return batch, label

    def parse_layers(self, layers_str):
        layers_list_str = layers_str.split(',')

        layers_list = []
        for layer_str in layers_list_str:
            layer_dimension_list = []
            layer_dimension_list_str = layer_str.split('*')

            for layer_dimension_str in layer_dimension_list_str:
                layer_dimension_list.append(int(layer_dimension_str))

            layers_list.append(layer_dimension_list)

        return layers_list

    def calculate_num_of_weight(self, layers, pad=0, stride=1):
        layer_type = []
        num_of_weight_per_layer = []
        num_of_bias_per_layer = []
        num_of_neuron_per_layer = []
        num_of_neuron_per_layer_without_pool = []

        for layer in layers:
            if layer is layers[0]:
                type = 'input'  # input
                layer_type.append(type)
                num_of_neuron_per_layer.append(layer)
                num_of_neuron_per_layer_without_pool.append(layer)

            elif layer is layers[-1]:
                type = 'output'  # output, fully-connected
                layer_type.append(type)
                num_of_weight = np.prod(
                    layer)*np.prod(num_of_neuron_per_layer[-1])
                num_of_weight_per_layer.append(num_of_weight)
                num_of_bias_per_layer.append(np.prod(layer))
                num_of_neuron_per_layer.append(layer)
                num_of_neuron_per_layer_without_pool.append(layer)

            elif len(layer) == 4:
                type = 'conv'  # convolutional
                layer_type.append(type)

                num_of_weight_per_layer.append(np.prod(layer))
                num_of_bias_per_layer.append(layer[3])

                h = (num_of_neuron_per_layer[-1]
                     [0] - layer[0] + 2*pad) // stride + 1
                w = (num_of_neuron_per_layer[-1]
                     [1] - layer[1] + 2*pad) // stride + 1
                d = layer[3]

                max_pool_f = 2
                max_pool_stride = 2

                h_max_pool = (h - max_pool_f) // max_pool_stride + 1
                w_max_pool = (w - max_pool_f) // max_pool_stride + 1
                d_max_pool = d

                num_of_neuron_per_layer.append(
                    [h_max_pool, w_max_pool, d_max_pool])
                num_of_neuron_per_layer_without_pool.append([h, w, d])
                layer_type.append('max_pool')

            else:
                type = 'hidden'  # fully-connected
                layer_type.append(type)
                num_of_weight = np.prod(
                    layer)*np.prod(num_of_neuron_per_layer[-1])
                num_of_weight_per_layer.append(num_of_weight)
                num_of_bias_per_layer.append(np.prod(layer))
                num_of_neuron_per_layer.append(layer)
                num_of_neuron_per_layer_without_pool.append(layer)

        print('layer_type:', layer_type)
        print('num_of_neuron_per_layer:', num_of_neuron_per_layer)
        print('num_of_neuron_per_layer_without_pool:',
              num_of_neuron_per_layer_without_pool)
        print('num_of_weight_per_layer:', num_of_weight_per_layer)
        print('num_of_bias_per_layer:', num_of_bias_per_layer)

        return [layer_type, num_of_neuron_per_layer, num_of_neuron_per_layer_without_pool,
                num_of_weight_per_layer, num_of_bias_per_layer]

    def update_output_hessians(self, sess, graph, train_set, sample_size):
        print('update_output_hessians')

        # get tensors
        x = graph.get_tensor_by_name("neuron_0:0")
        y_ = graph.get_tensor_by_name("y_:0")
        keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        cross_entropy = graph.get_tensor_by_name("cross_entropy:0")

        neurons = []
        output = graph.get_tensor_by_name(self.neuron_base_name + '0:0')
        neurons.append(output)

        for layer_no in range(1, len(self.layers)):
            if len(self.layers[layer_no]) >= 4:  # conv
                output_name = self.output_base_name + str(layer_no)
            else:
                output_name = self.neuron_base_name + str(layer_no)
            output = graph.get_tensor_by_name(output_name + ':0')
            neurons.append(output)

        output_importance_sum = [None] * len(neurons)

        importance_lst = []
        for i in range(len(neurons)):
            gradient = tf.gradients(cross_entropy, neurons[i])
            hessian_approximate = tf.square(gradient[0])
            importance = hessian_approximate * tf.square(neurons[i])
            importance_lst.append(importance)

        n_train_samples = train_set[0].shape[0]
        indices = np.arange(n_train_samples)
        np.random.shuffle(indices)

        for k in range(sample_size):
            print('sample', k)

            input_data, labels = self.next_batch(train_set, 1, k, indices)
            input_data_reshaped = \
                np.reshape(input_data, ([-1] + x.get_shape().as_list()[1:]))

            for i in range(len(neurons)):
                """
                hessian = tf.hessians(cross_entropy, neurons[i])
                assert len(hessian) == 1
                importance = tf.diag_part(hessian[0]) * tf.square(neurons[i])
                print(importance)
                s = sess.run(importance, feed_dict={x: input_data_reshaped,
                        y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})
                """
                print(importance_lst[i])
                s = sess.run(importance_lst[i], feed_dict={x: input_data_reshaped,
                                                           y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})

                if output_importance_sum[i] is None:
                    output_importance_sum[i] = s
                else:
                    output_importance_sum[i] += s

        for i in range(len(output_importance_sum)):
            output_importance_sum[i] = np.squeeze(
                output_importance_sum[i], axis=0)
            if len(output_importance_sum[i].shape) == 3:
                output_importance_sum[i] = np.transpose(
                    output_importance_sum[i], (2, 0, 1))

        return output_importance_sum

    def compute_activation_magnitude(self, sess, graph, train_set, sample_size):
        print('compute_activation_magnitude')

        # get tensors
        x = graph.get_tensor_by_name("neuron_0:0")
        y_ = graph.get_tensor_by_name("y_:0")
        keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        neurons = []
        output = graph.get_tensor_by_name(self.neuron_base_name + '0:0')
        neurons.append(output)

        for layer_no in range(1, len(self.layers)):
            if len(self.layers[layer_no]) >= 4:  # conv
                output_name = self.output_base_name + str(layer_no)
            else:
                output_name = self.neuron_base_name + str(layer_no)
            output = graph.get_tensor_by_name(output_name + ':0')
            neurons.append(output)

        output_importance_sum = [None] * len(neurons)

        importance_lst = neurons

        n_train_samples = train_set[0].shape[0]
        indices = np.arange(n_train_samples)
        np.random.shuffle(indices)

        for k in range(sample_size):
            print('sample', k)

            input_data, labels = self.next_batch(train_set, 1, k, indices)
            input_data_reshaped = \
                np.reshape(input_data, ([-1] + x.get_shape().as_list()[1:]))

            output_importance = sess.run(importance_lst, feed_dict={x: input_data_reshaped,
                                                                    y_: labels,
                                                                    keep_prob_input: 1.0,
                                                                    keep_prob: 1.0})

            assert len(output_importance_sum) == len(output_importance)

            for i in range(len(neurons)):
                if output_importance_sum[i] is None:
                    output_importance_sum[i] = output_importance[i]
                else:
                    output_importance_sum[i] += output_importance[i]

        for i in range(len(output_importance_sum)):
            output_importance_sum[i] = np.squeeze(
                output_importance_sum[i], axis=0)
            if len(output_importance_sum[i].shape) == 3:
                output_importance_sum[i] = np.transpose(
                    output_importance_sum[i], (2, 0, 1))

        return output_importance_sum

    @staticmethod
    def save_importance(importance, importance_file_path):
        if not os.path.exists(importance_file_path):
            print('saving new importance')
            new_importance = importance
        else:
            print('updating the importance')
            old_importance = Network.load_importance(importance_file_path)
            for i in range(len(old_importance)):
                old_importance[i] = old_importance[i] + importance[i]
            new_importance = old_importance

        # Random
        # new_importance = [None] * len(importance)
        # for i in range(len(importance)):
        #     new_importance[i] = np.random.random(size=importance[i].shape)

        np.save(importance_file_path, new_importance, allow_pickle=True)

    @staticmethod
    def load_importance(importance_file_path):
        importance = np.load(importance_file_path, allow_pickle=True)
        return importance

    def compute_importance(self, dataset, sample_size):

        # Compute importance for all classes at once.
        print("Computing importance on all classes")
        train_set = dataset.train_set(class_id=None)
        validation_set = dataset.validation_set(class_id=None)
        self._compute_importance(
            train_set, validation_set, sample_size, self.importance_file_path)

        # Compute importance per class separately
        for class_id in range(dataset.NUM_CLASSES):
            print(f"Computing importance on class {class_id}")
            train_set = dataset.train_set(class_id=class_id)
            validation_set = dataset.validation_set(class_id=class_id)
            importance_file_path = os.path.join(self.network_dir,
                                                self.importance_file_name + f'_classid_{class_id}.npy')
            self._compute_importance(
                train_set, validation_set, sample_size, importance_file_path)

        return

    def _compute_importance(self, train_set, validation_set, sample_size, importance_file_path):
        print("compute_importance")
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                self.loadNetwork(sess)  # Load the entire DNN
                # importance = self.update_output_hessians(
                #     sess, graph, train_set, sample_size)  # Passes only the training data
                importance = self.compute_activation_magnitude(
                    sess, graph, train_set, sample_size)
                self.save_importance(importance, importance_file_path)


class SubFlowNetwork:
    def __init__(self, network_no, network, network_name=None):
        self.network = network
        self.network_no = network_no
        self.network_name = network_name
        self.layers = network.layers
        self.layer_type, self.num_of_neuron_per_layer, self.num_of_neuron_per_layer_without_pool, self.num_of_weight_per_layer,\
            self.num_of_bias_per_layer = self.calculate_num_of_weight(
                self.layers)

        self.num_of_neuron = 0
        for layer in self.num_of_neuron_per_layer:
            self.num_of_neuron += np.prod(layer)

        self.num_of_weight = sum(self.num_of_weight_per_layer)
        self.num_of_bias = sum(self.num_of_bias_per_layer)

        self.network_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        'sub_network' + str(self.network_no))
        self.network_file_name = 'sub_network' + str(self.network_no)
        self.network_file_path = os.path.join(
            self.network_dir, self.network_file_name)

        self.importance_file_name = 'importance'
        self.importance_file_path = os.path.join(
            self.network_dir, self.importance_file_name + '.npy')

        self.neuron_base_name = "neuron_"
        self.weight_base_name = "weight_"
        self.bias_base_name = "bias_"
        self.output_base_name = "output_"
        self.activation_base_name = "activation_"

        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                self.build_sub_network(sess)
                if not os.path.exists(self.network_dir):
                    os.makedirs(self.network_dir)
                self.loadParameter(sess)
                self.saveNetwork(sess)

    def build_sub_network(self, sess):
        layer_type = copy.deepcopy(self.layer_type)
        layer_type = list(filter(lambda type: type != 'max_pool', layer_type))
        layers = copy.deepcopy(self.layers)
        parameters = {}
        neurons = {}
        outputs = {}
        parameters_to_regularize = []
        activations = {}

        keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # :XXX. The actual full network has NHWC as input layer
        # and this has NCHW
        neurons[0] = tf.placeholder(tf.float32,
                                    [None, layers[0][2], layers[0]
                                        [0], layers[0][1]],
                                    name=self.neuron_base_name+'0')
        outputs[0] = tf.identity(neurons[0], name=self.neuron_base_name+'0')
        print(neurons[0])

        for layer_no in range(1, len(layers)):
            weight_name = self.weight_base_name + str(layer_no-1)
            bias_name = self.bias_base_name + str(layer_no-1)
            neuron_name = self.neuron_base_name + str(layer_no)
            output_name = self.output_base_name + str(layer_no)
            activation_name = self.activation_base_name + str(layer_no)
            output_dim = self.num_of_neuron_per_layer_without_pool[layer_no]

            if layer_type[layer_no] == "conv":

                input_dim = self.num_of_neuron_per_layer[layer_no-1]
                filter_dim = layers[layer_no]
                stride_dim = [1, 1, 1, 1]

                conv_table = self.get_conv_table(
                    input_dim[0:2], filter_dim[0:2], stride_dim[1:3])
                what_to_conv, where_to_conv, conv_len \
                    = self.get_what_where_to_conv(conv_table, input_dim[0:2],
                                                  filter_dim[0:2], stride_dim[1:3])

                conv_parameter = {
                    'weights': tf.get_variable(weight_name,
                                               shape=(layers[layer_no][3], layers[layer_no][2],
                                                      layers[layer_no][0], layers[layer_no][1]),
                                               initializer=tf.contrib.layers.xavier_initializer()),
                    'biases': tf.get_variable(bias_name,
                                              shape=(layers[layer_no][3]),
                                              initializer=tf.contrib.layers.xavier_initializer()),
                }

                parameters_to_regularize.append(tf.reshape(conv_parameter['weights'],
                                                           [tf.size(conv_parameter['weights'])]))
                parameters_to_regularize.append(tf.reshape(conv_parameter['biases'],
                                                           [tf.size(conv_parameter['biases'])]))

                parameters[layer_no-1] = conv_parameter
                print('conv_parameter', parameters[layer_no-1])

                rank = sess.run(tf.rank(neurons[layer_no-1]))

                for _ in range(4 - rank):
                    neurons[layer_no -
                            1] = tf.expand_dims(neurons[layer_no-1], -1)

                activation = tf.placeholder(tf.int32,
                                            [output_dim[2], output_dim[0],
                                                output_dim[1]],
                                            name=activation_name)
                activations[layer_no] = activation

                output = sub_conv2d_library.sub_conv(neurons[layer_no-1],
                                                     conv_parameter['weights'], stride_dim, what_to_conv,
                                                     where_to_conv, conv_len, activation)
                output_biased = tf.multiply(tf.nn.bias_add(output,
                                                           conv_parameter['biases'], data_format='NCHW'),
                                            tf.cast(activation, tf.float32))

                print('output_biased', output_biased)

                # max pooling
                k = 2
                activation = tf.nn.leaky_relu(output_biased, name=output_name)
                outputs[layer_no] = activation

                neuron = tf.nn.max_pool(activation,
                                        ksize=[1, 1, k, k], strides=[1, 1, k, k],
                                        padding='VALID', data_format='NCHW', name=neuron_name)

                neurons[layer_no] = neuron

            elif layer_type[layer_no] == "hidden" or layer_type[layer_no] == "output":
                fc_parameter = {
                    'weights': tf.get_variable(weight_name,
                                               shape=(np.prod(self.num_of_neuron_per_layer[layer_no-1]),
                                                      np.prod(self.num_of_neuron_per_layer[layer_no])),
                                               initializer=tf.contrib.layers.xavier_initializer()),
                    'biases': tf.get_variable(bias_name,
                                              shape=(
                                                  np.prod(self.num_of_neuron_per_layer[layer_no])),
                                              initializer=tf.contrib.layers.xavier_initializer()),
                }

                parameters_to_regularize.append(tf.reshape(fc_parameter['weights'],
                                                           [tf.size(fc_parameter['weights'])]))
                parameters_to_regularize.append(tf.reshape(fc_parameter['biases'],
                                                           [tf.size(fc_parameter['biases'])]))

                parameters[layer_no-1] = fc_parameter
                print('fc_parameter', parameters[layer_no-1])

                # fully-connected
                flattened = tf.reshape(neurons[layer_no-1],
                                       [-1, np.prod(self.num_of_neuron_per_layer[layer_no-1])])
                neuron_drop = tf.nn.dropout(flattened, rate=1 - keep_prob)

                activation = tf.placeholder(tf.int32, [output_dim[0]],
                                            name=activation_name)
                activations[layer_no] = activation

                if layer_type[layer_no] == "hidden":
                    neuron = tf.nn.leaky_relu(tf.add(sub_matmul_library.sub_matmul(neuron_drop,
                                                                                   fc_parameter['weights'], activation),
                                                     tf.multiply(fc_parameter['biases'], tf.cast(activation, tf.float32))),
                                              name=neuron_name)

                elif layer_type[layer_no] == "output":
                    y_b = tf.add(sub_matmul_library.sub_matmul(neuron_drop,
                                                               fc_parameter['weights'], activation), fc_parameter['biases'])
                    neuron = tf.divide(tf.exp(y_b-tf.reduce_max(y_b)),
                                       tf.reduce_sum(
                                           tf.exp(y_b-tf.reduce_max(y_b))),
                                       name=neuron_name)

                neurons[layer_no] = neuron
                outputs[layer_no] = tf.identity(neuron, name=output_name)
            print(neuron)

        # output
        y = neurons[len(layers)-1]

        # correct labels
        y_ = tf.placeholder(tf.float32, [None] + layers[-1], name='y_')

        # define the loss function
        regularization = 0.00001 * \
            tf.nn.l2_loss(tf.concat(parameters_to_regularize, 0))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                                      reduction_indices=[1]), name='cross_entropy') + regularization

        # define accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1),
                                      name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                  name='accuracy')

        # for training
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           name='optimizer').minimize(cross_entropy)

        init = tf.global_variables_initializer()
        sess.run(init)

    def loadNetwork(self, sess):
        saver = tf.train.import_meta_graph(self.network_file_path + '.meta')
        saver.restore(sess, self.network_file_path)

    def saveNetwork(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.network_file_path)

    def loadParameter(self, sess):
        trainable_varialbe = tf.trainable_variables()
        parameter = np.load(
            self.network.parameter_file_path, allow_pickle=True)

        assign_tensor = []
        for i in range(len(trainable_varialbe)):
            if len(parameter[i].shape) >= 4:
                parameter_t = np.transpose(parameter[i], (3, 2, 0, 1))
                assign = tf.assign(trainable_varialbe[i], parameter_t)
            else:
                assign = tf.assign(trainable_varialbe[i], parameter[i])
            assign_tensor.append(assign)

        sess.run(assign_tensor)
        saver = tf.train.Saver()
        saver.save(sess, self.network_file_path)

    def do_sub_train(self, sess, graph, train_set, validation_set, batch_size,
                     train_iteration, activation, epochs):
        print("do_sub_train")

        # get tensors
        x = graph.get_tensor_by_name(self.neuron_base_name+"0:0")
        y_ = graph.get_tensor_by_name("y_:0")
        keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        optimizer = graph.get_operation_by_name("optimizer")

        tensor_list = []

        for i in range(1, len(self.layers)):
            mask = graph.get_tensor_by_name(
                self.activation_base_name+str(i) + ":0")
            tensor_list.append(mask)

        tensor_list.append(keep_prob_input)
        tensor_list.append(keep_prob)

        data_list = activation[1:]
        data_list.append(1.0)  # keep_prob_input
        data_list.append(1.0)  # keep_prob

        # Reshape the input data into NCHW. The input could be in flattened or NHWC format which is required
        # for the full network. The layers are defined in NHWC.
        input_data_validation_reshaped = np.reshape(
            validation_set[0], ([-1] + self.layers[0]))
        # Transpose to convert it to NCHW
        input_data_validation_reshaped = np.transpose(
            input_data_validation_reshaped, (0, 3, 1, 2))

        labels_validation = validation_set[1]

        # train
        n_train_samples = train_set[0].shape[0]
        train_iteration = (n_train_samples // batch_size)
        print(f"do_sub_train: Total iterations = {epochs*train_iteration}")
        for epoch in range(epochs):
            indices = np.arange(n_train_samples)
            np.random.shuffle(indices)
            for i in range(train_iteration):
                input_data, labels = self.next_batch(
                    train_set, batch_size, i, indices)
                input_data_reshaped = \
                    np.reshape(
                        input_data, ([-1] + self.layers[0]))
                input_data_reshaped = np.transpose(
                    input_data_reshaped, (0, 3, 1, 2))

                tensor_feed = tensor_list + [x, y_]
                data_feed = data_list + [input_data_reshaped, labels]
                sess.run(optimizer, feed_dict={
                    t: d for t, d in zip(tensor_feed, data_feed)})

            tensor_feed = tensor_list + [x, y_]
            data_feed = data_list + \
                [input_data_validation_reshaped, labels_validation]
            test_accuracy = sess.run(accuracy, feed_dict={
                t: d for t, d in zip(tensor_feed, data_feed)})
            print("epoch %d, Validation accuracy: %f" %
                  (epoch, test_accuracy))
            self.saveNetwork(sess)

    def get_activation(self, utilization, importance, random_mask=False):
        def sorted_indices(importance, num_of_active_neurons):
            # ascending sort indices
            arg_sort = np.argsort(importance.ravel())
            # descending sort indeces
            arg_sort = arg_sort[::-1]
            arg_sort = arg_sort[:num_of_active_neurons]
            return arg_sort

        def random_indices(importance, num_of_active_neurons):
            length = np.prod(importance.shape)
            indices = np.random.choice(
                np.arange(length), num_of_active_neurons, replace=False)
            return indices

        print("get_activation")

        len_active_output = 0
        len_output = 0
        activation_list = []

        for i in range(0, len(importance)):
            if i == 0:
                activation = np.ones_like(importance[i])
            elif i == len(importance)-1:
                activation = np.ones_like(importance[i])
            else:
                length = np.prod(importance[i].shape)
                activation = np.zeros(length, dtype=np.int32)
                num_of_active_output = int(np.floor(length * utilization))
                if not random_mask:
                    indices = sorted_indices(
                        importance[i], num_of_active_output)
                else:
                    indices = random_indices(
                        importance[i], num_of_active_output)
                activation[indices] = 1
                activation = np.reshape(activation, importance[i].shape)
                len_active_output += num_of_active_output
                len_output += length

            activation_list.append(activation)

        print('utilization %f' % utilization)

        return activation_list

    def sub_train(self, train_set, validation_set, batch_size,
                  train_iteration, utilization, epochs):
        print("sub_train")
        importance = Network.load_importance(self.network.importance_file_path)
        for i in range(len(utilization)):
            activation = self.get_activation(utilization[i], importance)
            with tf.Graph().as_default() as graph:
                with tf.Session(graph=graph) as sess:
                    self.loadNetwork(sess)
                    self.do_sub_train(sess, graph, train_set, validation_set, batch_size,
                                      train_iteration, activation, utilization[i], epochs)

    def do_sub_infer(self, sess, graph, data_set, activation,
                     utilization, label=None):
        tensor_x_name = "neuron_0:0"
        x = graph.get_tensor_by_name(tensor_x_name)
        tensor_y_name = "neuron_" + str(len(self.layers)-1) + ":0"
        y = graph.get_tensor_by_name(tensor_y_name)
        keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        tensor_list = []

        for i in range(1, len(self.layers)):
            mask = graph.get_tensor_by_name(
                self.activation_base_name+str(i) + ":0")
            tensor_list.append(mask)

        tensor_list.append(keep_prob_input)
        tensor_list.append(keep_prob)

        data_list = activation[1:]
        data_list.append(1.0)  # keep_prob_input
        data_list.append(1.0)  # keep_prob

        # infer
        data_set_reshaped = np.reshape(data_set, ([-1] + self.layers[0]))
        data_set_reshaped = np.transpose(data_set_reshaped, (0, 3, 1, 2))

        print(
            f"sub_infer, Input data shape {x.get_shape().as_list()}, reshaped data shape {data_set_reshaped.shape}")
        tensor_feed = tensor_list + [x]
        data_feed = data_list + [data_set_reshaped]

        infer_result = sess.run(y, feed_dict={
            t: d for t, d in zip(tensor_feed, data_feed)})

        test_accuracy = None
        if label is not None:
            predicted_samples = 0
            total_samples = label.shape[0]
            for pred, gt in zip(infer_result, label):
                if np.argmax(pred) == np.argmax(gt):
                    predicted_samples += 1
            test_accuracy = predicted_samples / total_samples
            print("Validation Inference accuracy:%f=%f, (pred/total)=(%d/%d)" %
                  (utilization, test_accuracy, predicted_samples, total_samples))

        return infer_result, test_accuracy

    def sub_infer(self, dataset, utilization, use_class_importance):
        # Compute inference per class separately
        acc_per_class = []
        samples_per_class = []
        for class_id in range(dataset.NUM_CLASSES):
            print(f"Computing inference on class {class_id}")
            test_set = dataset.test_set(class_id=class_id)
            if use_class_importance:
                importance_file_path = os.path.join(self.network.network_dir,
                                                    self.network.importance_file_name + f'_classid_{class_id}.npy')
            else:
                importance_file_path = self.network.importance_file_path

            _, test_accuracy = self._sub_infer(test_set[0], utilization=utilization, label=test_set[1],
                                               importance_file_path=importance_file_path)
            print("Class %d validation inference accuracy:%f=%f" %
                  (class_id, utilization, test_accuracy))
            acc_per_class.append(test_accuracy)
            samples_per_class.append(test_set[0].shape[0])

        test_accuracy = np.average(acc_per_class, weights=samples_per_class)
        print("All classes overall validation inference accuracy:%f=%f" %
              (utilization, test_accuracy))
        return

    def _sub_infer(self, data_set, utilization, label, importance_file_path):
        print("sub_infer")
        importance = Network.load_importance(importance_file_path)
        activation = self.get_activation(utilization, importance)

        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                self.loadNetwork(sess)
                return self.do_sub_infer(sess, graph, data_set,
                                         activation, utilization, label)

    def next_batch(self, data_set, batch_size, batch_index, indices):
        data = data_set[0]
        label = data_set[1]  # one-hot vectors

        start_index = batch_index*batch_size
        end_index = start_index + batch_size
        data_num = indices[start_index: end_index]

        batch = data[data_num, :]
        label = label[data_num, :]  # one-hot vectors

        return batch, label

    def parse_layers(self, layers_str):
        layers_list_str = layers_str.split(',')

        layers_list = []
        for layer_str in layers_list_str:
            layer_dimension_list = []
            layer_dimension_list_str = layer_str.split('*')

            for layer_dimension_str in layer_dimension_list_str:
                layer_dimension_list.append(int(layer_dimension_str))

            layers_list.append(layer_dimension_list)

        return layers_list

    def calculate_num_of_weight(self, layers, pad=0, stride=1):
        layer_type = []
        num_of_weight_per_layer = []
        num_of_bias_per_layer = []
        num_of_neuron_per_layer = []
        num_of_neuron_per_layer_without_pool = []

        for layer in layers:
            if layer is layers[0]:
                type = 'input'  # input
                layer_type.append(type)
                num_of_neuron_per_layer.append(layer)
                num_of_neuron_per_layer_without_pool.append(layer)

            elif layer is layers[-1]:
                type = 'output'  # output, fully-connected
                layer_type.append(type)
                num_of_weight = np.prod(
                    layer)*np.prod(num_of_neuron_per_layer[-1])
                num_of_weight_per_layer.append(num_of_weight)
                num_of_bias_per_layer.append(np.prod(layer))
                num_of_neuron_per_layer.append(layer)
                num_of_neuron_per_layer_without_pool.append(layer)

            elif len(layer) == 4:
                type = 'conv'  # convolutional
                layer_type.append(type)

                num_of_weight_per_layer.append(np.prod(layer))
                num_of_bias_per_layer.append(layer[3])

                h = (num_of_neuron_per_layer[-1]
                     [0] - layer[0] + 2*pad) // stride + 1
                w = (num_of_neuron_per_layer[-1]
                     [1] - layer[1] + 2*pad) // stride + 1
                d = layer[3]

                max_pool_f = 2
                max_pool_stride = 2

                h_max_pool = (h - max_pool_f) // max_pool_stride + 1
                w_max_pool = (w - max_pool_f) // max_pool_stride + 1
                d_max_pool = d

                num_of_neuron_per_layer.append(
                    [h_max_pool, w_max_pool, d_max_pool])
                num_of_neuron_per_layer_without_pool.append([h, w, d])
                layer_type.append('max_pool')

            else:
                type = 'hidden'  # fully-connected
                layer_type.append(type)
                num_of_weight = np.prod(
                    layer)*np.prod(num_of_neuron_per_layer[-1])
                num_of_weight_per_layer.append(num_of_weight)
                num_of_bias_per_layer.append(np.prod(layer))
                num_of_neuron_per_layer.append(layer)
                num_of_neuron_per_layer_without_pool.append(layer)

        print('layer_type:', layer_type)
        print('num_of_neuron_per_layer:', num_of_neuron_per_layer)
        print('num_of_neuron_per_layer_without_pool:',
              num_of_neuron_per_layer_without_pool)
        print('num_of_weight_per_layer:', num_of_weight_per_layer)
        print('num_of_bias_per_layer:', num_of_bias_per_layer)

        return [layer_type, num_of_neuron_per_layer, num_of_neuron_per_layer_without_pool,
                num_of_weight_per_layer, num_of_bias_per_layer]

    def get_conv_table(self, input_dim, filter_dim, stride_dim):

        output_height = (input_dim[0] - filter_dim[0]) // stride_dim[0] + 1
        output_width = (input_dim[1] - filter_dim[1]) // stride_dim[1] + 1
        output_dim = [output_height, output_width]

        conv_table = np.full((np.prod(input_dim), np.prod(output_dim)),
                             0, dtype=np.int32)

        filter_sequence = np.arange(1, np.prod(filter_dim)+1)

        start_row = -1
        for col in range(conv_table.shape[1]):
            start_row += 1
            if col/output_dim[1] > 0 and col % output_dim[1] == 0:
                start_row += (filter_dim[1] - 1)

            row = start_row
            for i in range(len(filter_sequence)):
                if i/filter_dim[1] > 0 and i % filter_dim[1] == 0:
                    row += (input_dim[1] - filter_dim[1])
                conv_table[row][col] = filter_sequence[i]
                row += 1

        assert np.count_nonzero(conv_table) == np.prod(
            filter_dim)*np.prod(output_dim)
        return conv_table

    def get_what_where_to_conv(self, conv_table, input_dim, filter_dim, stride_dim):
        conv_len = np.count_nonzero(conv_table, axis=1)
        output_height = (input_dim[0] - filter_dim[0]) // stride_dim[0] + 1
        output_width = (input_dim[1] - filter_dim[1]) // stride_dim[1] + 1
        output_dim = [output_height, output_width]

        where_to_conv = []
        where_to = []
        where = np.argwhere(conv_table > 0)

        for i in range(len(where)):
            if i == 0:
                where_to.append(where[i][1])
            else:
                if where[i][0] != where[i-1][0]:
                    where_to_conv += list(copy.deepcopy(where_to))
                    del where_to[:]
                where_to.append(where[i][1])

            if i == len(where)-1:
                where_to_conv += list(where_to)

        what_to_conv = []
        for i in range(conv_table.shape[0]):
            what_to = np.asarray(conv_table[i, :][conv_table[i, :] > 0]) - 1
            assert len(what_to) == conv_len[i]
            what_to_conv += list(what_to)

        return what_to_conv, where_to_conv, conv_len


def run_alexnet():
    model = models.AlexMaskNet(1000)

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load weight parameters
        model.load_parameters(sess)

        # model.compute_importance(sess=sess, sample_size=100000)
        model.validate_accuracy(sess)


def main(args):
    ng = None
    if os.path.exists(save_filename):
        ng = pickle.load(open(save_filename, 'rb'))
    else:
        ng = SubFlow()

    data = None
    if args.data is not None and args.data != '':
        data = __import__(args.data)

    if args.mode == 'l':
        print('[l] Task')
        for network in ng.network_list:
            print(network)
            print('\tnetwork_no:', network[1].network_no)
            print('\tnetwork_name:', network[1].network_name)
            print('\tnetwork_file_path:', network[1].network_file_path)
            print('\tlayers(%d):' % len(network[1].layers), network[1].layers)
            print('\tlayer_type(%d):'
                  % len(network[1].layer_type), network[1].layer_type)
            print('\tnum_of_neuron_per_layer(%d):'
                  % network[1].num_of_neuron, network[1].num_of_neuron_per_layer)
            print('\tnum_of_neuron_per_layer_without_pool(%d):'
                  % network[1].num_of_neuron,
                  network[1].num_of_neuron_per_layer_without_pool)
            print('\tnum_of_weight_per_layer(%d):'
                  % network[1].num_of_weight, network[1].num_of_weight_per_layer)
            print('\tnum_of_bias_per_layer(%d):'
                  % network[1].num_of_bias, network[1].num_of_bias_per_layer)

        return

    elif args.mode == 'c':
        print('[c] constructing a network')

        if args.layers == None or args.layers == '':
            print('[c] No layer. Use -layers')
            return

        print('[c] layers:', args.layers)
        ng.constructNetwork(args.layers, args.name)
        # Uncomment the following line to test alexnet.
        # Lazy to add another command line argument.
        # run_alexnet()

    elif args.mode == 'd':
        print('[d] destructing a network')
        if args.network_no == -1:
            print('[d] No network_no. Use -network_no')
            return

        ng.destructNetwork(args.network_no)

    elif args.mode == 't':
        print('[t] train')
        if args.network_no == -1:
            print('[t] No network_no. Use -network_no')
            return

        if data == None:
            print('[t] No data. Use -data')
            return

        print('[t] network_no:', args.network_no)
        print('[t] data:', args.data,
              'train/test.size:', data.train_set()[0].shape, data.test_set()[0].shape)

        batch_size = 96
        train_iteration = 4000

        for network in ng.network_list:
            if network[0] == args.network_no:
                network[1].train(data.train_set(), data.validation_set(),
                                 batch_size, train_iteration, args.epochs)

    elif args.mode == 'i':
        print('[i] inference')
        if args.network_no == -1:
            print('[i] No network_no. Use -network_no')
            return

        if data == None:
            print('[i] No data. Use -data')
            return

        for network in ng.network_list:
            if network[0] == args.network_no:
                network[1].infer(data.test_set()[0], data.test_set()[1])
                return

        print("no network", args.network_no)
        return

    elif args.mode == 'ci':
        print('[ci] compute importance')
        if args.network_no == -1:
            print('[ci] No network_no. Use -network_no')
            return

        if data == None:
            print('[ci] No data. Use -data')
            return

        print('[ci] network_no:', args.network_no)
        print('[ci] data:', args.data,
              'train/test.size:', data.train_set()[0].shape, data.test_set()[0].shape)
        print('[ci] use_class_importance:', args.use_class_importance)

        import calendar
        ts = calendar.timegm(time.gmtime())
        seed = ts % 10000
        np.random.seed(seed)

        for network in ng.network_list:
            if network[0] == args.network_no:
                network[1].compute_importance(data, sample_size=100)

    elif args.mode == 'sc':
        print('[sc] constructing a sub_network')

        if args.network_no == None or args.network_no == '':
            print('[sc] No network_no. Use -network_no')
            return

        network = None
        for networks in ng.network_list:
            if networks[0] == args.network_no:
                network = networks[1]
                break

        if not network:
            print('[sc] invalid network_no', args.network_no)
            return

        print('[sc] network_no:', args.network_no)

        ng.construct_sub_network(network, args.name)

    elif args.mode == 'sd':
        print('[sd] destructing a sub_network')
        if args.subflow_network_no == -1:
            print('[sd] No subflow_network_no. Use -subflow_network_no')
            return

        ng.destruct_sub_network(args.subflow_network_no)

    elif args.mode == 'st':
        print('[st] sub_train')
        if args.subflow_network_no == -1:
            print('[st] No subflow_network_no. Use -subflow_network_no')
            return

        if args.utilization == 0:
            import calendar
            ts = calendar.timegm(time.gmtime())
            seed = ts % 10000
            np.random.seed(seed)

            utilization = []
            for i in np.arange(1.0, 0.0, -0.1):
                utilization.extend([i] * 10)
        else:
            # utilization = [args.utilization] * args.epochs
            utilization = [args.utilization]

        if data == None:
            print('[st] No data. Use -data')
            return

        print('[st] subflow_network_no:', args.subflow_network_no)
        print('[st] data:', args.data,
              'train/test.size:', data.train_set()[0].shape, data.test_set()[0].shape)

        batch_size = 96
        train_iteration = 100

        for sub_network in ng.sub_network_list:
            if sub_network[0] == args.subflow_network_no:
                sub_network[1].sub_train(data.train_set(), data.test_set(),
                                         batch_size, train_iteration, utilization,
                                         epochs=args.epochs)

    elif args.mode == 'si':
        print('[si] sub_inference')
        if args.subflow_network_no == -1:
            print('[si] No subflow_network_no. Use -subflow_network_no')
            return

        if args.utilization == 0:
            import calendar
            ts = calendar.timegm(time.gmtime())
            seed = ts % 10000
            np.random.seed(seed)
            utilization = np.arange(0.1, 1.1, 0.1)
        else:
            utilization = [args.utilization]

        if data == None:
            print('[si] No data. Use -data')
            return

        for sub_network in ng.sub_network_list:
            if sub_network[0] == args.subflow_network_no:
                for i in range(len(utilization)):
                    sub_network[1].sub_infer(data, utilization[i],
                                             use_class_importance=args.use_class_importance)
                return

        print("no sub_network", args.subflow_network_no)
        return

    if args.save != False:
        pickle.dump(ng, open(save_filename, 'wb'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-mode', type=str,	help='mode', default='l')
    # l: show the current status of SubFlow (-mode=l)
    # c: construct a network (-mode=c -data -layers)
    # d: destruct a network
    # t: train a network
    # i: inference of a network

    parser.add_argument('-layers', type=str, help='layers', default=None)
    # Conv layer: 4 dimensions (height, width, depth, num of filters) [ex. 3,3,3,8]
    # Fully-connected layer: any dimension less than 4 [ex. 128 / 128*128 / 128*128*1]

    parser.add_argument('-data', type=str, help='data', default=None)
    parser.add_argument('-network_no', type=int, help='network_no', default=-1)
    parser.add_argument('-subflow_network_no', type=int,
                        help='subflow_network_no', default=-1)
    parser.add_argument('-utilization', type=float,
                        help='utilization', default=1.0)
    parser.add_argument('-name', type=str, help='name', default=None)
    parser.add_argument('-save', type=bool, help='save SubFlow?', default=True)
    parser.add_argument('-epochs', type=int,
                        help='SubFlow training epochs', default=1)
    parser.add_argument('-use_class_importance', type=int, default=0, choices=[0, 1],
                        help="Integer flag to use per class neuron importance")

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
