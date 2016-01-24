import tensorflow as tf

from collections import namedtuple


Topology = namedtuple(
    "Topology",
    "x y t w_hidden b_hidden hidden w_out b_out")


def generate_weight(shape, name):
    return tf.Variable(
        tf.random_normal(
            shape,
            stddev=0.01,
            dtype=tf.float32), name=name)


def build(num_hidden, num_features, num_labels):
    x = tf.placeholder(tf.float32, shape=[None, num_features])
    t = tf.placeholder(tf.float32, shape=[None, num_labels])

    w_hidden = generate_weight([num_features, num_hidden], "w_hidden")
    b_hidden = generate_weight([1, num_hidden], "b_hidden")

    hidden = tf.nn.relu(tf.matmul(x, w_hidden) + b_hidden)

    w_out = generate_weight([num_hidden, num_labels], "w_out")
    b_out = generate_weight([1, num_labels], "b_out")

    y = tf.nn.softmax(tf.matmul(hidden, w_out) + b_out, name="y")

    return Topology(
        x=x,
        y=y,
        t=t,
        w_hidden=w_hidden,
        b_hidden=b_hidden,
        hidden=hidden,
        w_out=w_out,
        b_out=b_out)
