import tensorflow as tf
import os
import numpy as np

from imagenet_data import NUM_CLASSES, dataset_generator


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

        # Build network
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                self.fc8 = self.build_network()
                self.load_parameters(sess, self.fc8, weights_file)
                # self.save_network(sess)
                self.validate_accuracy(sess)

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

    def pool_layer(prev_layer, layer_no, ksize, strides=1, max_pool=True):
        if max_pool:
            output = tf.nn.max_pool(prev_layer,
                                    ksize=[1, ksize, ksize, 1],
                                    strides=[1, strides, strides, 1],
                                    name=f"max_pool_{layer_no}",
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
                output, keep_prob=keep_prob, name=f"fc_dropout_{layer_no}")

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

        return fc8

    def load_parameters(self, sess, fc8,  weights_file):
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
        batch_size = 1
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

        sess.run(tf.global_variables_initializer())

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
