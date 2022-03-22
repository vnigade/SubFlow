import tensorflow as tf
import os
import numpy as np
from imagenet_data import NUM_CLASSES, dataset_generator
import copy
import time


class AlexNet:
    """ This code is inspired and also borrowed from.

        https://github.com/kratzert/finetune_alexnet_with_tensorflow
    """

    NETWORK_NAME = "AlexNet"

    def __init__(self, num_classes=1000, weights_file="./weights/bvlc_alexnet.npy") -> None:
        self.num_classes = num_classes
        self.network_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        'networks')
        self.network_file_name = 'network_' + self.NETWORK_NAME
        self.network_file_path = os.path.join(
            self.network_dir, self.network_file_name)

        self.parameter_file_name = 'parameter_' + self.NETWORK_NAME
        self.parameter_file_path = os.path.join(
            self.network_dir, self.parameter_file_name + '.npy')

        self.importance_file_name = 'importance'
        self.importance_file_path = os.path.join(
            self.network_dir, self.importance_file_name + '.npy')

        self._neuron_names_lst = []

        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                # Build network
                self.fc8 = self.build_network()

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                # Load weight parameters
                self.load_parameters(sess, weights_file)

                # self.validate_accuracy(sess)
                # self.save_network(sess)

                # compute importance
                self.compute_importance(sess, graph)

    def conv_layer(prev_layer, filter_height, filter_width, num_filters, padding, layer_no, strides=1, groups=1):
        channels = int((prev_layer.shape)[-1])
        shape = [filter_height, filter_width,
                 int(channels/groups), num_filters]
        with tf.variable_scope(f"conv{layer_no}") as scope:
            conv_parameter = {
                'weights': tf.get_variable(f"weights",
                                           shape=shape,
                                           initializer=tf.contrib.layers.xavier_initializer()),
                'biases': tf.get_variable(f"biases",
                                          shape=[num_filters],
                                          initializer=tf.contrib.layers.xavier_initializer()),
            }

        if groups == 1:
            output = tf.nn.conv2d(prev_layer,
                                  conv_parameter['weights'],
                                  strides=[1, strides, strides, 1],
                                  padding=padding,
                                  name=f"conv2d_{layer_no}")
        else:
            input_groups = tf.split(
                axis=3, num_or_size_splits=groups, value=prev_layer)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                     value=conv_parameter['weights'])
            output_groups = [tf.nn.conv2d(i, k,
                                          strides=[1, strides, strides, 1],
                                          padding=padding,
                                          name=f"conv2d_{layer_no}")
                             for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            output = tf.concat(axis=3, values=output_groups)

        output_biased = tf.reshape(tf.nn.bias_add(
            output, conv_parameter['biases'], name=f"bias_add_{layer_no}"), tf.shape(output))

        # default relu
        activation = tf.nn.relu(
            output_biased, name=f"conv_relu_{layer_no}")

        return activation

    def pool_layer(prev_layer, layer_no, ksize, strides=1, data_format="NHWC", max_pool=True):
        if max_pool:
            output = tf.nn.max_pool(prev_layer,
                                    ksize=[1, ksize, ksize, 1],
                                    strides=[1, strides, strides, 1],
                                    name=f"max_pool_{layer_no}",
                                    data_format=data_format,
                                    padding='VALID')
        return output

    def fc_layer(prev_layer, layer_no, num_input, num_output, relu=True, dropout=False, keep_prob=None):
        with tf.variable_scope(f"fc{layer_no}") as scope:
            fc_parameter = {
                'weights': tf.get_variable(f"weights",
                                           shape=(num_input, num_output),
                                           initializer=tf.contrib.layers.xavier_initializer()),
                'biases': tf.get_variable(f"biases",
                                          shape=(num_output),
                                          initializer=tf.contrib.layers.xavier_initializer()),
            }
        output = tf.nn.xw_plus_b(
            prev_layer, fc_parameter["weights"], fc_parameter["biases"], name=f"fc_{layer_no}")

        if relu:
            output = tf.nn.relu(output, name=f"fc_relu_{layer_no}")

        if dropout:
            output = tf.nn.dropout(
                output, rate=(1 - keep_prob), name=f"fc_dropout_{layer_no}")

        return output

    def lrn(prev_layer, layer_no, radius=2, alpha=2e-05, beta=0.75, bias=1.0):
        output = tf.nn.local_response_normalization(prev_layer, depth_radius=radius,
                                                    alpha=alpha, beta=beta,
                                                    bias=bias, name=f"lrn_{layer_no}")
        return output

    def build_network(self):
        # Inputs
        self._input = tf.placeholder(tf.float32, [None] + [227, 227, 3],
                                     name='input_0')
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        layer_no = 1
        conv1 = AlexNet.conv_layer(prev_layer=self._input,
                                   filter_height=11,
                                   filter_width=11,
                                   num_filters=96,
                                   strides=4,
                                   padding="VALID", layer_no=layer_no)

        lrn1 = AlexNet.lrn(conv1, layer_no=layer_no)
        pool1 = AlexNet.pool_layer(
            prev_layer=lrn1, ksize=3, strides=2, layer_no=layer_no)

        layer_no += 1
        conv2 = AlexNet.conv_layer(prev_layer=pool1,
                                   filter_height=5,
                                   filter_width=5,
                                   num_filters=256,
                                   strides=1, groups=2,
                                   padding="SAME", layer_no=layer_no)
        lrn2 = AlexNet.lrn(conv2, layer_no=layer_no)
        pool2 = AlexNet.pool_layer(
            prev_layer=lrn2, ksize=3, strides=2, layer_no=layer_no)

        layer_no += 1
        conv3 = AlexNet.conv_layer(prev_layer=pool2,
                                   filter_height=3,
                                   filter_width=3,
                                   num_filters=384,
                                   strides=1,
                                   padding="SAME", layer_no=layer_no)

        layer_no += 1
        conv4 = AlexNet.conv_layer(prev_layer=conv3,
                                   filter_height=3,
                                   filter_width=3,
                                   num_filters=384,
                                   strides=1, groups=2,
                                   padding="SAME", layer_no=layer_no)

        layer_no += 1
        conv5 = AlexNet.conv_layer(prev_layer=conv4,
                                   filter_height=3,
                                   filter_width=3,
                                   num_filters=256,
                                   strides=1, groups=2,
                                   padding="SAME", layer_no=layer_no)
        pool5 = AlexNet.pool_layer(
            prev_layer=conv5, ksize=3, strides=2, layer_no=layer_no)

        # FC layers
        shape = pool5.shape
        neurons = int(shape[1]) * int(shape[2]) * int(shape[3])
        flattened_pool = tf.reshape(pool5, [-1, neurons])

        layer_no += 1
        fc6 = AlexNet.fc_layer(flattened_pool, num_input=neurons, num_output=4096,
                               dropout=True, keep_prob=self._keep_prob, layer_no=layer_no)

        layer_no += 1
        fc7 = AlexNet.fc_layer(fc6, num_input=4096, num_output=4096,
                               dropout=True, keep_prob=self._keep_prob, layer_no=layer_no)

        layer_no += 1
        fc8 = AlexNet.fc_layer(fc7, num_input=4096,
                               num_output=self.num_classes,
                               relu=False,
                               dropout=False, layer_no=layer_no)

        # self._neuron_names_lst = [self._input.name, conv1.name,
        #                           conv2.name, conv3.name, conv4.name, conv5.name,
        #                           fc6.name, fc7.name, fc8.name]

        self._neuron_names_lst = [conv1.name, conv2.name, conv3.name, conv4.name, conv5.name,
                                  fc6.name, fc7.name]

        return fc8

    def load_parameters(self, sess,  weights_file):
        """Load weights from file into network.
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # weights_file = "/home/vinod/remote_files/das5/scratch/packages/SubFlow/weights/bvlc_alexnet.npy"
        # Load the weights into memory
        weights_dict = np.load(
            weights_file, allow_pickle=True, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            with tf.variable_scope(op_name, reuse=True):
                # Assign weights/biases to their corresponding tf variable
                for data in weights_dict[op_name]:
                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable=False)
                        sess.run(var.assign(data))

                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        sess.run(var.assign(data))

    def load_network(self, sess):
        saver = tf.train.import_meta_graph(self.network_file_path + '.meta')
        saver.restore(sess, self.network_file_path)

    def save_network(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.network_file_path)
        parameter = sess.run(tf.trainable_variables())
        np.save(self.parameter_file_path, parameter, allow_pickle=True)

    def train_network(self):
        pass

    def validate_accuracy(self, sess):
        batch_size = 4
        data_dir = "/var/scratch/mreisser/imagenet/ILSVRC2012_img_val_tf_records"
        dataset = dataset_generator(
            data_dir, type="validation", batch_size=batch_size)

        iter = dataset.make_one_shot_iterator()
        next_batch = iter.get_next()

        # validation_init_op = iter.make_initializer()

        # TF placeholder for graph input and output
        self._y = tf.placeholder(tf.float32, [batch_size, NUM_CLASSES])

        score = self.fc8

        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(self._y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        test_count = 0
        test_acc = 1
        try:

            while True:
                x_batch, y_batch, filename_batch = sess.run(next_batch)
                print(filename_batch)
                acc, y_score = sess.run([accuracy, score], feed_dict={self._input: x_batch,
                                                                      self._y: y_batch,
                                                                      self._keep_prob: 1.})

                print(f"Shape: gt: {y_batch.shape}, pred: {y_score.shape}")
                print(f"gt: {np.argmax(y_batch)}, pred: {np.argmax(y_score)}")
                test_acc += acc
                test_count += 1
        except tf.errors.OutOfRangeError:
            pass
        except Exception as e:
            print(f"Exception {str(e)}")

        test_acc /= test_count
        print("Validation Accuracy = {:.4f}".format(test_acc))

    def save_importance(self, importance):
        if not os.path.exists(self.importance_file_path):
            print('saving new importance')
            new_importance = importance
        else:
            print('updating the importance')
            old_importance = np.load(
                self.importance_file_path, allow_pickle=True)
            for i in range(len(old_importance)):
                old_importance[i] = old_importance[i] + importance[i]
            new_importance = old_importance

        # for i in range(len(new_importance)):
        #	print(new_importance[i])

        np.save(self.importance_file_path, new_importance, allow_pickle=True)

    def compute_importance(self, sess, graph, sample_size=1000):
        batch_size = 1
        input = self._input
        output_score = self.fc8
        labels_gt = tf.placeholder(tf.float32, [batch_size, NUM_CLASSES])

        # Define cross entropy
        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=output_score,
                                                        labels=labels_gt))

        # Get the list of input, conv, fc and output layer
        neurons_lst = []
        for neuron_names in self._neuron_names_lst:
            neuron = graph.get_tensor_by_name(neuron_names)
            neurons_lst.append(neuron)

        data_dir = "/var/scratch/mreisser/imagenet/ILSVRC2012_img_val_tf_records"
        dataset = dataset_generator(
            data_dir, type="train", batch_size=batch_size)

        iter = dataset.make_one_shot_iterator()
        next_batch = iter.get_next()

        output_importance_sum = [None] * len(neurons_lst)
        try:
            for k in range(sample_size):
                x_batch, y_batch, filename_batch = sess.run(next_batch)
                print(filename_batch)

                for i in range(len(neurons_lst)):
                    gradient = tf.gradients(cross_entropy, neurons_lst[i])
                    hessian_approximate = tf.square(gradient[0])
                    importance = hessian_approximate * \
                        tf.square(neurons_lst[i])

                    print(importance)
                    s = sess.run(importance, feed_dict={self._input: x_batch,
                                                        labels_gt: y_batch,
                                                        self._keep_prob: 1.})

                    if output_importance_sum[i] is None:
                        output_importance_sum[i] = s
                    else:
                        output_importance_sum[i] += s
        except tf.errors.OutOfRangeError:
            pass
        except Exception as e:
            print(f"Exception {str(e)}")

        for i in range(len(output_importance_sum)):
            output_importance_sum[i] = np.squeeze(
                output_importance_sum[i], axis=0)
            if len(output_importance_sum[i].shape) == 3:
                print(output_importance_sum[i].shape)
                # @TODO: Very if we need to transpose.
                # This is the input or conv layer.
                # The data_format is 'NHWC'. Therefore, we don't have to transpose.
                # But if data_format is 'NCHW' uncomment the following code statement.
                # output_importance_sum[i] = np.transpose(
                #     output_importance_sum[i], (2, 0, 1))


class AlexMaskNet:

    NETWORK_NAME = "AlexMaskNet"

    def __init__(self, num_classes=1000, weights_file="./weights/bvlc_alexnet.npy") -> None:
        self.num_classes = num_classes
        self._weights_file = weights_file
        self.network_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        'networks', self.NETWORK_NAME)
        if not os.path.exists(self.network_dir):
            os.makedirs(self.network_dir)

        self.network_file_name = 'network_' + self.NETWORK_NAME
        self.network_file_path = os.path.join(
            self.network_dir, self.network_file_name)

        self.parameter_file_name = 'parameter_' + self.NETWORK_NAME
        self.parameter_file_path = os.path.join(
            self.network_dir, self.parameter_file_name + '.npy')

        self.importance_file_name = 'importance'
        self.importance_file_path = os.path.join(
            self.network_dir, self.importance_file_name + '.npy')

        self._neuron_names_lst = []
        self._activation_mask_lst = []

        # Build network
        self.fc8 = self.build_network()

    def conv_layer(self, prev_layer, filter_height, filter_width, num_filters, output_dim, padding, layer_no, strides=1, groups=1):
        channels = int((prev_layer.shape)[-1])
        shape = [filter_height, filter_width,
                 int(channels/groups), num_filters]
        with tf.variable_scope(f"conv{layer_no}") as scope:
            conv_parameter = {
                'weights': tf.get_variable(f"weights",
                                           shape=shape,
                                           initializer=tf.contrib.layers.xavier_initializer()),
                'biases': tf.get_variable(f"biases",
                                          shape=[num_filters],
                                          initializer=tf.contrib.layers.xavier_initializer()),
            }

        if groups == 1:
            output = tf.nn.conv2d(prev_layer,
                                  conv_parameter['weights'],
                                  strides=[1, strides, strides, 1],
                                  padding=padding,
                                  name=f"conv2d_{layer_no}")
        else:
            input_groups = tf.split(
                axis=3, num_or_size_splits=groups, value=prev_layer)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                     value=conv_parameter['weights'])
            output_groups = [tf.nn.conv2d(i, k,
                                          strides=[1, strides, strides, 1],
                                          padding=padding,
                                          name=f"conv2d_{layer_no}")
                             for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            output = tf.concat(axis=3, values=output_groups)

        output_biased = tf.reshape(tf.nn.bias_add(
            output, conv_parameter['biases'], name=f"bias_add_{layer_no}"), tf.shape(output))

        # default relu
        activation = tf.nn.relu(
            output_biased, name=f"conv_relu_{layer_no}")

        # apply activation mask
        activation_mask = tf.placeholder(tf.int32, shape=output_dim,
                                         name=f"conv{layer_no}_activation_mask")
        self._activation_mask_lst.append(activation_mask)
        activation = tf.multiply(activation,
                                 tf.cast(activation_mask, tf.float32))

        return activation

    def fc_layer(self, prev_layer, layer_no, num_input, num_output, relu=True, dropout=False, keep_prob=None):
        with tf.variable_scope(f"fc{layer_no}") as scope:
            fc_parameter = {
                'weights': tf.get_variable(f"weights",
                                           shape=(num_input, num_output),
                                           initializer=tf.contrib.layers.xavier_initializer()),
                'biases': tf.get_variable(f"biases",
                                          shape=(num_output),
                                          initializer=tf.contrib.layers.xavier_initializer()),
            }

        output = tf.nn.xw_plus_b(
            prev_layer, fc_parameter["weights"], fc_parameter["biases"], name=f"fc_{layer_no}")

        if relu:
            output = tf.nn.relu(output, name=f"fc_relu_{layer_no}")

        if dropout:
            output = tf.nn.dropout(
                output, rate=(1 - keep_prob), name=f"fc_dropout_{layer_no}")

        # apply activation mask
        activation_mask = tf.placeholder(tf.int32, shape=[num_output],
                                         name=f"fc{layer_no}_activation_mask")
        self._activation_mask_lst.append(activation_mask)
        activation = tf.multiply(output,
                                 tf.cast(activation_mask, tf.float32))

        return activation

    def build_network(self):
        # Inputs
        self._input = tf.placeholder(tf.float32, [None] + [227, 227, 3],
                                     name='input_0')
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        layer_no = 1
        conv1 = self.conv_layer(prev_layer=self._input,
                                filter_height=11,
                                filter_width=11,
                                num_filters=96,
                                strides=4,
                                output_dim=[55, 55, 96],
                                padding="VALID", layer_no=layer_no)

        lrn1 = AlexNet.lrn(conv1, layer_no=layer_no)
        pool1 = AlexNet.pool_layer(
            prev_layer=lrn1, ksize=3, strides=2, layer_no=layer_no)

        layer_no += 1
        conv2 = self.conv_layer(prev_layer=pool1,
                                filter_height=5,
                                filter_width=5,
                                num_filters=256,
                                strides=1, groups=2,
                                output_dim=[27, 27, 256],
                                padding="SAME", layer_no=layer_no)
        lrn2 = AlexNet.lrn(conv2, layer_no=layer_no)
        pool2 = AlexNet.pool_layer(
            prev_layer=lrn2, ksize=3, strides=2, layer_no=layer_no)

        layer_no += 1
        conv3 = self.conv_layer(prev_layer=pool2,
                                filter_height=3,
                                filter_width=3,
                                num_filters=384,
                                strides=1,
                                output_dim=[13, 13, 384],
                                padding="SAME", layer_no=layer_no)

        layer_no += 1
        conv4 = self.conv_layer(prev_layer=conv3,
                                filter_height=3,
                                filter_width=3,
                                num_filters=384,
                                strides=1, groups=2,
                                output_dim=[13, 13, 384],
                                padding="SAME", layer_no=layer_no)

        layer_no += 1
        conv5 = self.conv_layer(prev_layer=conv4,
                                filter_height=3,
                                filter_width=3,
                                num_filters=256,
                                strides=1, groups=2,
                                output_dim=[13, 13, 256],
                                padding="SAME", layer_no=layer_no)
        pool5 = AlexNet.pool_layer(
            prev_layer=conv5, ksize=3, strides=2, layer_no=layer_no)

        # FC layers
        shape = pool5.shape
        neurons = int(shape[1]) * int(shape[2]) * int(shape[3])
        flattened_pool = tf.reshape(pool5, [-1, neurons])

        layer_no += 1
        fc6 = self.fc_layer(flattened_pool, num_input=neurons, num_output=4096,
                            dropout=True, keep_prob=self._keep_prob, layer_no=layer_no)

        layer_no += 1
        fc7 = self.fc_layer(fc6, num_input=4096, num_output=4096,
                            dropout=True, keep_prob=self._keep_prob, layer_no=layer_no)

        layer_no += 1
        fc8 = self.fc_layer(fc7, num_input=4096,
                            num_output=self.num_classes,
                            relu=False,
                            dropout=False, layer_no=layer_no)

        # self._neuron_names_lst = [self._input.name, conv1.name,
        #                           conv2.name, conv3.name, conv4.name, conv5.name,
        #                           fc6.name, fc7.name, fc8.name]
        self._neuron_names_lst = [conv1.name,
                                  conv2.name, conv3.name, conv4.name, conv5.name,
                                  fc6.name, fc7.name]

        return fc8

    def load_parameters(self, sess):
        """Load weights from file into network.
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(
            self._weights_file, allow_pickle=True, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            with tf.variable_scope(op_name, reuse=True):
                # Assign weights/biases to their corresponding tf variable
                for data in weights_dict[op_name]:
                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable=False)
                        sess.run(var.assign(data))

                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        sess.run(var.assign(data))

    def load_network(self, sess):
        saver = tf.train.import_meta_graph(self.network_file_path + '.meta')
        saver.restore(sess, self.network_file_path)

    def save_network(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.network_file_path)
        parameter = sess.run(tf.trainable_variables())
        np.save(self.parameter_file_path, parameter, allow_pickle=True)

    def train_network(self):
        pass

    def _validate_accuracy(self, sess, importance, utilization):
        batch_size = 8
        data_dir = "/var/scratch/mreisser/imagenet/ILSVRC2012_img_val_tf_records"
        dataset = dataset_generator(
            data_dir, type="validation", batch_size=batch_size)

        iter = dataset.make_one_shot_iterator()
        next_batch = iter.get_next()

        # validation_init_op = iter.make_initializer()

        # TF placeholder for graph input and output
        self._y = tf.placeholder(tf.float32, [batch_size, NUM_CLASSES])
        score = self.fc8

        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(self._y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tensor_feed = [self._input, self._y, self._keep_prob]
        tensor_feed = tensor_feed + self._activation_mask_lst

        test_count = 0
        test_acc = 1

        RANDOM_MASK = True
        activation_masks = self.get_activation_mask(
            importance, utilization, random_mask=RANDOM_MASK)

        try:
            while True:
                x_batch, y_batch, filename_batch = sess.run(next_batch)
                if RANDOM_MASK:
                    activation_masks = self.get_activation_mask(
                        importance, utilization, random_mask=RANDOM_MASK)
                # print(filename_batch)
                tensor_values = [x_batch, y_batch, 1.0]
                tensor_values = tensor_values + activation_masks
                acc, y_score = sess.run([accuracy, score], feed_dict={
                    t: v for t, v in zip(tensor_feed, tensor_values)})

                # print(f"Shape: gt: {y_batch.shape}, pred: {y_score.shape}")
                # for i in range(batch_size):
                #     print(
                #         f"gt: {np.argmax(y_batch[i])}, pred: {np.argmax(y_score[i])}")
                test_acc += acc
                test_count += 1
        except tf.errors.OutOfRangeError:
            pass
        except Exception as e:
            print(f"Exception {str(e)}")

        test_acc /= test_count
        print("Validation Accuracy = {:.4f}".format(test_acc))
        return test_acc

    def save_importance(self, importance):
        if not os.path.exists(self.importance_file_path):
            print('saving new importance')
            new_importance = importance
        else:
            print('updating the importance')
            old_importance = np.load(
                self.importance_file_path, allow_pickle=True)
            for i in range(len(old_importance)):
                old_importance[i] = old_importance[i] + importance[i]
            new_importance = old_importance

        # for i in range(len(new_importance)):
        #	print(new_importance[i])

        np.save(self.importance_file_path, new_importance, allow_pickle=True)

    def compute_importance(self, sess,  sample_size=1000):
        batch_size = 1
        input = self._input
        output_score = self.fc8
        labels_gt = tf.placeholder(tf.float32, [batch_size, NUM_CLASSES])

        # Define cross entropy
        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=output_score,
                                                        labels=labels_gt))

        # Get the list of input, conv, fc and output layer
        graph = tf.get_default_graph()
        neurons_lst = []
        for neuron_names in self._neuron_names_lst:
            neuron = graph.get_tensor_by_name(neuron_names)
            neurons_lst.append(neuron)

        activation_mask = []
        for i, tensor in enumerate(self._activation_mask_lst):
            shape = tensor.shape
            activation_mask.append(np.ones(shape=shape))

        tensor_feed = [self._input, labels_gt, self._keep_prob]
        tensor_feed = tensor_feed + self._activation_mask_lst

        data_dir = "/var/scratch/mreisser/imagenet/ILSVRC2012_img_val_tf_records"
        dataset = dataset_generator(
            data_dir, type="train", batch_size=batch_size)

        iter = dataset.make_one_shot_iterator()
        next_batch = iter.get_next()

        importance_lst = []
        for neuron in neurons_lst:
            gradient = tf.gradients(cross_entropy, neuron)
            hessian_approximate = tf.square(gradient[0])
            importance = hessian_approximate * tf.square(neuron)
            importance_lst.append(importance)
        output_importance_sum = [None] * len(neurons_lst)

        try:
            for k in range(sample_size):
                t1 = time.time()
                x_batch, y_batch, filename_batch = sess.run(next_batch)
                t2 = time.time()
                tensor_values = [x_batch, y_batch, 1.0]
                tensor_values = tensor_values + activation_mask

                print(filename_batch)

                t3 = time.time()
                for i in range(len(neurons_lst)):
                    # gradient = tf.gradients(cross_entropy, neurons_lst[i])
                    # hessian_approximate = tf.square(gradient[0])
                    # importance = hessian_approximate * \
                    #     tf.square(neurons_lst[i])

                    # print(importance_lst[i])
                    s = sess.run(importance_lst[i], feed_dict={
                        t: v for t, v in zip(tensor_feed, tensor_values)})

                    # print(s)
                    if output_importance_sum[i] is None:
                        output_importance_sum[i] = s
                    else:
                        output_importance_sum[i] += s
                t4 = time.time()

                print(f"{k}/{sample_size}\nSample loading time: {(t2 - t1) * 1e3}, "
                      f" Importance computation: {(t4 - t3) * 1e3}")
        except tf.errors.OutOfRangeError:
            pass
        except Exception as e:
            print(f"Exception {str(e)}")

        for i in range(len(output_importance_sum)):
            output_importance_sum[i] = np.squeeze(
                output_importance_sum[i], axis=0)
            if len(output_importance_sum[i].shape) == 3:
                print(output_importance_sum[i].shape)
                # @TODO: Very if we need to transpose.
                # This is the input or conv layer.
                # The data_format is 'NHWC'. Therefore, we don't have to transpose.
                # But if data_format is 'NCHW' uncomment the following code statement.
                # output_importance_sum[i] = np.transpose(
                #     output_importance_sum[i], (2, 0, 1))

        np.save(self.importance_file_path,
                output_importance_sum, allow_pickle=True)

    def get_activation_mask(self, importance, utilization, random_mask=False):
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

        total_active_neurons = 0
        total_neurons = 0
        activation_list = []
        # for i, tensor in enumerate(self._activation_mask_lst):
        #     shape = tensor.shape
        #     activation_list.append(np.ones(shape=shape))

        # return activation_list

        # It does not include the input and output layer.
        for i in range(0, len(importance)):
            length = np.prod(importance[i].shape)
            activation = np.zeros(length, dtype=np.int32)
            num_of_active_neurons = int(np.floor(length * utilization))
            if not random_mask:
                indices = sorted_indices(importance[i], num_of_active_neurons)
            else:
                indices = random_indices(importance[i], num_of_active_neurons)
            activation[indices] = 1
            activation = np.reshape(activation, importance[i].shape)
            total_active_neurons += num_of_active_neurons
            total_neurons += length

            activation_list.append(activation)

        # For last output layer
        for i, tensor in enumerate(self._activation_mask_lst):
            if i >= len(activation_list):
                shape = tensor.shape
                activation_list.append(np.ones(shape=shape))

        real_utilization = float(total_active_neurons) / total_neurons
        # print('utilization %f' % utilization)

        return activation_list

    def validate_accuracy(self, sess):
        importance = np.load(self.importance_file_path, allow_pickle=True)
        # importance = []
        acc_per_utilization = {}
        for i in range(1, 11):
            utilization = 0.1 * i
            print(f"Validating accuracy for utilization {utilization}")
            acc = self._validate_accuracy(sess, importance, utilization)
            acc_per_utilization[utilization] = acc

        for util, acc in acc_per_utilization.items():
            print(util, acc)


class AlexSubNet:
    NETWORK_NAME = "AlexSubNet"

    def __init__(self, sub_conv2d_library, sub_matmul_library,
                 num_classes=1000, weights_file="./weights/bvlc_alexnet.npy",
                 ) -> None:

        self.sub_conv2d_library = sub_conv2d_library
        self.sub_matmul_library = sub_matmul_library

        self.num_classes = num_classes
        self.network_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        'networks')
        self.network_file_name = 'network_' + self.NETWORK_NAME
        self.network_file_path = os.path.join(
            self.network_dir, self.network_file_name)

        self.parameter_file_name = 'parameter_' + self.NETWORK_NAME
        self.parameter_file_path = os.path.join(
            self.network_dir, self.parameter_file_name + '.npy')

        self.importance_file_name = 'importance'
        self.importance_file_path = os.path.join(
            self.network_dir, self.importance_file_name + '.npy')

        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                # Build network
                self.fc8 = self.build_network()

                # Initialize all variables
                # sess.run(tf.global_variables_initializer())

                # Load weight parameters
                # self.load_parameters(sess, weights_file)

                # self.validate_accuracy(sess)
                # self.save_network(sess)

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

    def conv_layer(self, prev_layer, filter_height, filter_width, num_filters, output_dim, padding, layer_no, strides, groups=1):
        # prev_layer might have data_format of NCHW. TODO: check it.
        channels = int((prev_layer.shape)[1])
        shape = [filter_height, filter_width,
                 int(channels/groups), num_filters]
        re_shape = [num_filters, int(
            channels/groups), filter_height, filter_width]

        with tf.variable_scope(f"conv{layer_no}") as scope:
            conv_parameter = {
                'weights': tf.get_variable(f"weights",
                                           shape=re_shape,
                                           initializer=tf.contrib.layers.xavier_initializer()),
                'biases': tf.get_variable(f"biases",
                                          shape=[num_filters],
                                          initializer=tf.contrib.layers.xavier_initializer()),
            }

        input_dim = [int(prev_layer.shape[2]), int(prev_layer.shape[3])]
        filter_dim = [filter_height, filter_width]
        stride_dim = [1, strides, strides, 1]

        conv_table = self.get_conv_table(
            input_dim, filter_dim, [strides, strides])
        what_to_conv, where_to_conv, conv_len \
            = self.get_what_where_to_conv(conv_table, input_dim, filter_dim, [strides, strides])

        activation = tf.placeholder(tf.int32,
                                    [output_dim[2], output_dim[0],
                                     output_dim[1]],
                                    name=f"conv{layer_no}_activation")

        if groups == 1:
            output = self.sub_conv2d_library.sub_conv(prev_layer,
                                                      conv_parameter['weights'], stride_dim,
                                                      what_to_conv, where_to_conv, conv_len, activation)
        else:
            input_groups = tf.split(
                axis=1, num_or_size_splits=groups, value=prev_layer)
            weight_groups = tf.split(axis=1, num_or_size_splits=groups,
                                     value=conv_parameter['weights'])
            output_groups = [self.sub_conv2d_library.sub_conv(i,
                                                              k, stride_dim,
                                                              what_to_conv, where_to_conv, conv_len, activation)
                             for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            output = tf.concat(axis=1, values=output_groups)

        output_biased = tf.multiply(tf.nn.bias_add(output, conv_parameter['biases'],
                                                   data_format='NCHW'),
                                    tf.cast(activation, tf.float32))

        output_biased = tf.reshape(output_biased, tf.shape(output))

        # default relu
        activation = tf.nn.relu(
            output_biased, name=f"conv_relu_{layer_no}")

        return activation

    def fc_layer(self, prev_layer, layer_no, num_input, num_output, relu=True, dropout=False, keep_prob=None):
        with tf.variable_scope(f"fc{layer_no}") as scope:
            fc_parameter = {
                'weights': tf.get_variable(f"weights",
                                           shape=(num_input, num_output),
                                           initializer=tf.contrib.layers.xavier_initializer()),
                'biases': tf.get_variable(f"biases",
                                          shape=(num_output),
                                          initializer=tf.contrib.layers.xavier_initializer()),
            }
        output = tf.nn.xw_plus_b(
            prev_layer, fc_parameter["weights"], fc_parameter["biases"], name=f"fc_{layer_no}")
        activation = tf.placeholder(tf.int32, [num_output],
                                    name=f"activation_fc{layer_no}")

        output = tf.add(self.sub_matmul_library.sub_matmul(prev_layer,
                                                           fc_parameter['weights'], activation),
                        tf.multiply(fc_parameter['biases'], tf.cast(
                            activation, tf.float32)))

        if relu:
            output = tf.nn.relu(output, name=f"fc_relu_{layer_no}")

        if dropout:
            output = tf.nn.dropout(
                output, rate=(1 - keep_prob), name=f"fc_dropout_{layer_no}")

        return output

    def build_network(self):
        # Inputs
        self._input = tf.placeholder(tf.float32, [None] + [3, 227, 227],
                                     name='input_0')
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        layer_no = 1
        conv1 = self.conv_layer(prev_layer=self._input,
                                filter_height=11,
                                filter_width=11,
                                num_filters=96,
                                strides=4,
                                output_dim=[55, 55, 96],
                                padding="VALID", layer_no=layer_no)

        lrn1 = AlexNet.lrn(conv1, layer_no=layer_no)
        pool1 = AlexNet.pool_layer(
            prev_layer=lrn1, ksize=3, strides=2, data_format='NCHW', layer_no=layer_no)

        layer_no += 1
        conv2 = self.conv_layer(prev_layer=pool1,
                                filter_height=5,
                                filter_width=5,
                                num_filters=256,
                                strides=1, groups=2,
                                output_dim=[27, 27, 256],
                                padding="SAME", layer_no=layer_no)
        lrn2 = AlexNet.lrn(conv2, layer_no=layer_no)
        pool2 = AlexNet.pool_layer(
            prev_layer=lrn2, ksize=3, strides=2, data_format='NCHW', layer_no=layer_no)

        layer_no += 1
        conv3 = self.conv_layer(prev_layer=pool2,
                                filter_height=3,
                                filter_width=3,
                                num_filters=384,
                                strides=1,
                                output_dim=[13, 13, 384],
                                padding="SAME", layer_no=layer_no)

        layer_no += 1
        conv4 = self.conv_layer(prev_layer=conv3,
                                filter_height=3,
                                filter_width=3,
                                num_filters=384,
                                strides=1, groups=2,
                                output_dim=[13, 13, 384],
                                padding="SAME", layer_no=layer_no)

        layer_no += 1
        conv5 = self.conv_layer(prev_layer=conv4,
                                filter_height=3,
                                filter_width=3,
                                num_filters=256,
                                strides=1, groups=2,
                                output_dim=[13, 13, 256],
                                padding="SAME", layer_no=layer_no)
        pool5 = AlexNet.pool_layer(
            prev_layer=conv5, ksize=3, strides=2, data_format='NCHW', layer_no=layer_no)

        # FC layers
        shape = pool5.shape
        neurons = int(shape[1]) * int(shape[2]) * int(shape[3])
        flattened_pool = tf.reshape(pool5, [-1, neurons])

        layer_no += 1
        fc6 = self.fc_layer(flattened_pool, num_input=neurons, num_output=4096,
                            dropout=True, keep_prob=self._keep_prob, layer_no=layer_no)

        layer_no += 1
        fc7 = self.fc_layer(fc6, num_input=4096, num_output=4096,
                            dropout=True, keep_prob=self._keep_prob, layer_no=layer_no)

        layer_no += 1
        fc8 = self.fc_layer(fc7, num_input=4096,
                            num_output=self.num_classes,
                            relu=False,
                            dropout=False, layer_no=layer_no)

        return fc8
