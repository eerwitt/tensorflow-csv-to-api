from iris import dataset
from iris import log

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Download the Iris flower dataset from UCI to be used in
            training a feed forward neural network.""")
    parser.add_argument(
        "--raw-dir",
        default="./raw",
        type=str,
        help="Location to download the Iris flower dataset to.")
    parser.add_argument(
        "--test-dir",
        default="./test",
        type=str,
        help="Location to place the test dataset generated from raw data.")
    parser.add_argument(
        "--train-dir",
        default="./train",
        type=str,
        help="Location to place the train dataset generated from raw data.")
    parser.add_argument(
        "--use-backup-data-url",
        action="store_true",
        help="""Use an alternate location (other than UCI) to download
            IRIS data from.""")
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    args = parser.parse_args()

    log.set_verbosity(args.verbosity)

    dataset.prepare(
        args.raw_dir,
        args.test_dir,
        args.train_dir,
        args.use_backup_data_url)
