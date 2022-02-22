import tensorflow as tf
from tensorflow.python.platform import gfile
import os

# tf.enable_eager_execution()

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
NUM_CLASSES = 1000


def _tf_record_parser(serialized_example):
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/channels': tf.FixedLenFeature([], tf.int64),
                                           'image/width': tf.FixedLenFeature([], tf.int64),
                                           'image/class/label': tf.FixedLenFeature([], tf.int64),
                                           'image/format': tf.FixedLenFeature([], tf.string),
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/class/synset': tf.FixedLenFeature([], tf.string),
                                           'image/height': tf.FixedLenFeature([], tf.int64),
                                           'image/class/text': tf.FixedLenFeature([], tf.string),
                                           'image/colorspace': tf.FixedLenFeature([], tf.string),
                                           'image/filename': tf.FixedLenFeature([], tf.string),
                                       })

    height = tf.cast(features['image/height'], tf.float32)
    width = tf.cast(features['image/width'], tf.float32)
    channels = tf.cast(features['image/channels'], tf.float32)
    img_size = tf.stack([height, width, channels], axis=0)
    label = features['image/class/label'] - 1

    img = features['image/encoded']
    img = tf.image.decode_jpeg(img, channels=3)

    return (img, label, features["image/filename"])


def _preprocessing(serialized_example, resize_image_size):
    img, label, filename = _tf_record_parser(serialized_example)
    img_resized = tf.image.resize_images(img, resize_image_size)
    img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
    # RGB -> BGR
    img_bgr = img_centered[:, :, ::-1]

    label_onehot = tf.one_hot(label, NUM_CLASSES)
    print(str(filename))
    return img_bgr, label_onehot, filename


def dataset_generator(data_dir: str, type: str, resize_image_size=[227, 227], batch_size=1):
    def map_fn(serialized_example):
        return _preprocessing(serialized_example, resize_image_size)

    glob_pattern = os.path.join(data_dir, f'{type}-*-of-*')
    file_names = gfile.Glob(glob_pattern)
    if not file_names:
        print("No files found")
        return

    # print(file_names)

    dataset = tf.data.TFRecordDataset(
        filenames=file_names)
    dataset = dataset.map(map_func=map_fn, num_parallel_calls=16)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


if __name__ == "__main__":
    data_dir = "/var/scratch/mreisser/imagenet/ILSVRC2012_img_val_tf_records"
    dataset = dataset_generator(
        data_dir, type="validation", batch_size=2)

    iter = dataset.make_one_shot_iterator()
    validation_init_op = iter.make_initializer()
    next_element = iter.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                x_batch, y_batch, filename_batch = sess.run(next_element)
                print(filename_batch)
        except tf.errors.OutOfRangeError:
            pass
        except Exception as e:
            print(f"Exception {str(e)}")
