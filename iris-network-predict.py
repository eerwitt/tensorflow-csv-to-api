import argparse

from iris import network
from iris import log


_logger = log.get_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Predict the species of an Iris based on its elements""")

    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        type=str,
        help="Location to restore checkpoint files.")
    parser.add_argument(
        "--feature",
        required=True,
        help="""
            CSV list with 4 elements as sepal_length, sepal_width,
            petal_length and petal_width e.g.
                1.5,2.3,4.5,6.7""",
        type=lambda a: [float(l) for l in a.split(",")])
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    args = parser.parse_args()

    log.set_verbosity(args.verbosity)

    features = [args.feature]
    y = network.predict(features, args.checkpoint_dir)

    for i in range(len(features)):
        feature = features[i]
        confidence = [round(p, 2) for p in y[0][i]]
        _logger.info(
            "Prediction for %s is %s with y of: %s",
            feature,
            y[1][i],
            confidence)
