import argparse

from iris import network
from iris import log


_logger = log.get_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Use the test dataset to check for the total accuracy of
            the highest iteration's checkpoint model.""")

    parser.add_argument(
        "--test-dir",
        default="./test",
        type=str,
        help="Directory containing CSV files used in testing.")
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        type=str,
        help="Location to restore checkpoint files.")
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    args = parser.parse_args()

    log.set_verbosity(args.verbosity)

    test_features, test_species = network.read_data_set(
        "{test_dir}/*.csv".format(test_dir=args.test_dir))

    network.test(
        test_features,
        test_species,
        args.checkpoint_dir)
