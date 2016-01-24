import tensorflow as tf

from collections import namedtuple
import glob

from iris import log

# Total guess based on the rule of thumb from this question:
# http://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde
NUM_HIDDEN = 21

# Sepal Length, Sepal Width, Petal Length and Petal Width == 4 features.
NUM_FEATURES = 4

# Iris-setosa, Iris-versicolor and Iris-virginica == 3
NUM_LABELS = 3

_logger = log.get_logger()

# Data structure used to keep track of important parameters used in training our
# model.
Topology = namedtuple(
    "Topology",
    "x y t w_hidden b_hidden hidden w_out b_out")


def generate_weight(shape, name):
    """
    TF variable filled with random values which follow a normal distribution.

    Parameters
    ----------
    shape : 1-D Tensor or Array
        Corresponds to the shape parameter of tf.random_normal.
    name : str
        Variable name used in saving and restoring checkpoints.

    Returns
    -------
    variable : tf.Variable
        TensorFlow variable which is filled with random values in the shape of
        the parameter "shape".

    Notes
    -----
    See also:
        https://en.wikipedia.org/wiki/Normal_distribution

    Originally found as part of:
        https://github.com/nlintz/TensorFlow-Tutorials/blob/master/3_net.py#L7
    """
    return tf.Variable(
        tf.random_normal(
            shape,
            stddev=0.01,
            dtype=tf.float32), name=name)


def build():
    """
    Create the topology for a feed forward neural network, we use this to
    declare shared functionality between the train, test and predict functions.

    Returns
    -------
    topology : Topology
        Set of variables which are required in training, testing and storing
        this model in checkpoint files.

    Notes
    -----
    See also:
        https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html#train-the-model
    """
    num_hidden = NUM_HIDDEN
    num_features = NUM_FEATURES
    num_labels = NUM_LABELS

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


def _split_model_name_to_negative_numeric(model_name):
    """
    Change a model file name into the integer value at the end of the filename.

    Parameters
    ----------
    model_name : str
        A model filename from the checkpoints directory.

    Returns
    -------
    model_number : int
        Negative number which relates to the number in the filename.

    Note
    ----
    Returns a negative number since it's used in sorting and we want to sort
    from highest to lowest.

    Example
    -------
    >>> model_name = "model-2000"
    >>> _split_model_name_to_negative_numeric(model_name)
    -2000
    """
    return -int(model_name.split('-')[1])


def most_recent_checkpoint(checkpoint_directory):
    """
    Sort the checkpoint files found under the checkpoints directory and return
    the highest numbered checkpoint file.

    Parameters
    ----------
    checkpoint_directory : str
        Directory including checkpoint model files.

    Returns
    -------
    checkpoint_filename : str
        Most recent checkpoint based on the numerical extension.
    """
    checkpoints = glob.glob("{dir}/model-*".format(dir=checkpoint_directory))

    return sorted(checkpoints, key=_split_model_name_to_negative_numeric)[0]


def read_data_set(directory):
    """
    Read CSV files from a directory using TensorFLow to generate the filenames
    and the queue of files to be processed.

    Parameters
    ----------
    directory : str
        A directory containing CSV files which match the format which has been
        pulled from our original CSV file.
        (sepal length, sepal width, petal length, petal width, iris species)

    Returns
    -------
    features, iris_species : tuple(array, scalar)
        The features pulled from the CSV and the target species.

    Notes
    -----
    This fails on TF 0.6 if you pass in a CSV without \n line endings.
    """
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
    """
    This code was copied between a script which uses a scoped session and a
    webserver which uses a session which is kept open until explicitly closed.

    Parameters
    ----------
    net : Topology
        Network to test against.
    features : Tensor
        Features used as the `x` while testing.
    sess : tf.Session
        An active session from TensorFlow, won't close after running the code.

    Returns
    -------
    prediction : Tensor(Tensor, Scalar)
        Prediction with raw `y` results and the species chosen based on the
        features. The argmax will return a number between 0 and NUM_LABELS which
        corresponds to where the 1 will be in our one-hot vector.
    """
    return sess.run(
        [net.y, tf.argmax(net.y, 1)],
        feed_dict={net.x: features})


def predict_init(checkpoint_dir):
    """
    Encapsulating shared logic between the script to run predictions and
    the web API. The logic is mainly related to initializing the variable and
    checkpoint for a network.

    Parameters
    ----------
    checkpoint_dir : str
        Directory which is used to store the checkpoint save files.

    Returns
    -------
    Tuple(tf.Session, tf.Tensor)
        Active TensorFlow session and a feed forward network.
    """
    net = build()

    checkpoint = tf.train.Saver([
        net.w_hidden, net.b_hidden, net.w_out, net.b_out])

    sess = tf.Session()
    checkpoint.restore(sess, most_recent_checkpoint(checkpoint_dir))

    return sess, net


def predict(features, checkpoint_dir):
    """
    Predict the species of an iris based on its features and the most recent
    checkpoint model.

    Parameters
    ----------
    features : tf.Tensor
        Meta data about an Iris to guess its species.
    checkpoint_dir : str
        Location of checkpoint model files.

    Returns
    -------
    prediction : tf.Tensor
        See #predict_with_session for the full return shape.
    """
    sess, net = predict_init(checkpoint_dir)

    prediction = predict_with_session(net, features, sess)

    sess.close()

    return prediction


def test(features, iris_species, checkpoint_dir):
    """
    Take a random test feature and attempt to predict its species. This is
    repeated 100x to get the accuracy of the trained model.

    Parameters
    ----------
    features : tf.Tensor
        Iris features without their species.
    iris_species : tf.Tensor
        One-hot vector of the resulting species for each feature.
    checkpoint_dir : str
        Relative location of all the model checkpoints.
    """
    net = build()

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

        checkpoint.restore(sess, most_recent_checkpoint(checkpoint_dir))

        total_accuracy = 0
        # We test 100 times 1-101 which pulls random elements from the test
        # dataset which has fewer than 100 items in it.
        for iteration in range(1, 101):
            example, label = sess.run([features, iris_species])
            label = [int(l) for l in label]

            current_accuracy = accuracy.eval(
                feed_dict={
                    net.x: [example],
                    net.t: [label]})
            total_accuracy += current_accuracy
            _logger.debug(
                "[example: %s, label: %s, accuracy: %d]",
                example,
                label,
                current_accuracy)

        _logger.info("Total Accuracy: %0.2d", total_accuracy)
        coord.request_stop()
        coord.join(threads)


def train(features, iris_species, checkpoint_dir, iterations, save_every):
    """
    Train a model based on a training set.

    Parameters
    ----------
    features : tf.Tensor
        Training features of Iris flowers found in the training set.
    iris_species : tf.Tensor
        Known species of Iris flowers described by features.
    checkpoint_dir : str
        Directory to save model checkpoint files to.
    iterations : int
        Maximum number of iterations (steps) to run before quitting training.
        Training continues to run until we tell it to stop...
    save_every : int
        Number of iterations between each save to a checkpoint model. Tweak this
        to stop from creating too many checkpoint files.

    Notes
    -----
    Logic for training and the GradientDescentOptimizer are taken from the
    training steps found at:
        https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/tf/index.html
    """
    net = build()

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
            example, label = sess.run([features, iris_species])
            label = [int(l) for l in label]

            train_step.run(feed_dict={
                net.x: [example], net.t: [label]})

            if iteration % save_every == 0:
                _logger.info("Saving iteration %i.", iteration)
                save_path = checkpoint.save(
                    sess,
                    "{cd}/model".format(cd=checkpoint_dir),
                    global_step=iteration)
                _logger.debug("File saved to %s", save_path)

        coord.request_stop()
        coord.join(threads)


def onehot_from_argmax(argmax_index):
    """
    Convert the argmax output to be a one-hot representation.

    Parameters
    ----------
    argmax_index : int{0..NUM_LABELS-1}
        Argmax output from finding the highest value in the Tensor from the
        prediction model.

    Returns
    -------
    onehot : str
        One-hot vector based on the highest index found in the y Tensor.
    """
    return "%0*d" % (NUM_LABELS, 10 ** (NUM_LABELS - argmax_index - 1))
