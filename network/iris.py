import tensorflow as tf
from tqdm import tqdm
import requests

import os.path
import glob

from network import topology


IRIS_DATA_URL = \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

NUM_HIDDEN = 50
NUM_FEATURES = 4
NUM_LABELS = 3


def download_iris_data(output_location):
    output_filename = "{ol}/iris-data-raw.csv".format(
        ol=output_location)
    if os.path.exists(output_filename):
        return False

    iris_data_response = requests.get(IRIS_DATA_URL, stream=True)
    with open(output_filename, "wb") as iris_data_output:
        for block in tqdm(iris_data_response.iter_content()):
            iris_data_output.write(block)

    return True


def _split_model_name_to_negative_numeric(model_name):
    return -int(model_name.split('-')[1])


def most_recent_checkpoint(directory):
    checkpoints = glob.glob("{dir}/model-*".format(dir=directory))

    return sorted(checkpoints, key=_split_model_name_to_negative_numeric)[0]


def read_data_set(directory):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(directory),
        shuffle=True)

    line_reader = tf.TextLineReader(skip_header_lines=1)

    _, csv_row = line_reader.read(filename_queue)

    record_defaults = [[0.0], [0.0], [0.0], [0.0], [""]]
    sepal_length, sepal_width, petal_length, petal_width, iris_species = \
        tf.decode_csv(csv_row, record_defaults=record_defaults)

    features = tf.pack([
        sepal_length,
        sepal_width,
        petal_length,
        petal_width])

    return features, iris_species


def predict_with_session(net, features, sess):
    return sess.run(
        [net.y, tf.argmax(net.y, 1)],
        feed_dict={net.x: features})


def predict_init(checkpoint_dir):
    num_hidden = NUM_HIDDEN
    num_features = NUM_FEATURES
    num_labels = NUM_LABELS

    net = topology.build(num_hidden, num_features, num_labels)

    checkpoint = tf.train.Saver([
        net.w_hidden, net.b_hidden, net.w_out, net.b_out])

    sess = tf.Session()
    checkpoint.restore(sess, most_recent_checkpoint(checkpoint_dir))

    return sess, net


def predict(features, checkpoint_dir):
    sess, net = predict_init(checkpoint_dir)

    prediction = predict_with_session(net, features, sess)

    sess.close()

    return prediction
