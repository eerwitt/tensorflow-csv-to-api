import tensorflow as tf

import argparse

from network import topology as tp
from network import iris
from network import log


logger = log.get_logger()


def train(features, iris_class, checkpoint_dir, iterations, save_every):
    num_hidden = iris.NUM_HIDDEN
    num_features = iris.NUM_FEATURES
    num_labels = iris.NUM_LABELS

    net = tp.build(num_hidden, num_features, num_labels)

    cross_entropy = -tf.reduce_sum(net.t * tf.log(net.y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    checkpoint = tf.train.Saver([
        net.w_hidden, net.b_hidden, net.w_out, net.b_out])
    with tf.Session() as sess:
        # Start populating the filename queue.
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if save_every is None:
            save_every = iterations / 10

        for iteration in range(1, iterations + 1):
            example, label = sess.run([features, iris_class])
            label = [int(l) for l in label]

            train_step.run(feed_dict={
                net.x: [example], net.t: [label]})

            if iteration % save_every == 0:
                logger.info("Saving iteration %i.", iteration)
                save_path = checkpoint.save(
                    sess,
                    "{cd}/model".format(cd=checkpoint_dir),
                    global_step=iteration)
                logger.debug("File saved to %s", save_path)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-dir',
        required=True,
        type=str,
        help="Directory containing CSV files used in training.")
    parser.add_argument(
        '--checkpoint-dir',
        required=True,
        type=str,
        help="Location to save checkpoint files.")
    parser.add_argument(
        '--checkpoint-save-every',
        type=int,
        help="Save a checkpoint every X iterations.")
    parser.add_argument(
        '--train-iterations',
        required=True,
        type=int,
        help="Number of train iterations to run.")
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    args = parser.parse_args()

    log.set_verbosity(args.verbosity)

    logger.info(
        "Reading CSV files from %s.", args.train_dir)

    train_features, train_class = iris.read_data_set(
        "{train_dir}/*.csv".format(train_dir=args.train_dir))

    train(
        train_features,
        train_class,
        args.checkpoint_dir,
        args.train_iterations,
        args.checkpoint_save_every)
