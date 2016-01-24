import argparse

from iris import network
from iris import log


_logger = log.get_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Train a neural network using the Iris dataset.""")

    parser.add_argument(
        "--train-dir",
        default="./train",
        type=str,
        help="Directory containing CSV files used in training.")
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        type=str,
        help="Location to save checkpoint files.")
    parser.add_argument(
        "--checkpoint-save-every",
        type=int,
        help="Save a checkpoint every X iterations.")
    parser.add_argument(
        "--train-iterations",
        default=20000,
        type=int,
        help="Number of train iterations to run.")
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    args = parser.parse_args()

    log.set_verbosity(args.verbosity)

    _logger.info(
        "Reading CSV files from %s.", args.train_dir)

    train_features, train_species = network.read_data_set(
        "{train_dir}/*.csv".format(train_dir=args.train_dir))

    network.train(
        train_features,
        train_species,
        args.checkpoint_dir,
        args.train_iterations,
        args.checkpoint_save_every)
