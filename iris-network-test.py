import tensorflow as tf

import argparse

from network import topology as tp
from network import iris
from network import log


logger = log.get_logger()


def test(features, iris_class, checkpoint_dir):
    num_hidden = iris.NUM_HIDDEN
    num_features = iris.NUM_FEATURES
    num_labels = iris.NUM_LABELS

    net = tp.build(num_hidden, num_features, num_labels)

    correct_prediction = tf.equal(
        tf.argmax(net.y, 1), tf.argmax(net.t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    checkpoint = tf.train.Saver([
        net.w_hidden, net.b_hidden, net.w_out, net.b_out])

    with tf.Session() as sess:
        # Start populating the filename queue.
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        checkpoint.restore(sess, iris.most_recent_checkpoint(checkpoint_dir))

        total_accuracy = 0
        for iteration in range(1, 101):
            example, label = sess.run([features, iris_class])
            label = [int(l) for l in label]

            current_accuracy = accuracy.eval(
                feed_dict={
                    net.x: [example],
                    net.t: [label]})
            total_accuracy += current_accuracy
            logger.debug(
                "[example: %s, label: %s, accuracy: %d]",
                example,
                label,
                current_accuracy)

        logger.info("Total Accuracy: %0.2dpct", total_accuracy)
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test-dir',
        required=True,
        type=str,
        help="Directory containing CSV files used in testing.")
    parser.add_argument(
        '--checkpoint-dir',
        required=True,
        type=str,
        help="Location to restore checkpoint files.")
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    args = parser.parse_args()

    log.set_verbosity(args.verbosity)

    test_features, test_class = iris.read_data_set(
        "{test_dir}/*.csv".format(test_dir=args.test_dir))

    test(
        test_features,
        test_class,
        args.checkpoint_dir)
