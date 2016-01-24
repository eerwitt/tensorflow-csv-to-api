import tensorflow as tf

import logging


# Default logger for keeping track of steps.
_logger = logging.getLogger("iris")

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
_logger.addHandler(_handler)


def get_logger():
    return _logger


def set_verbosity(verbosity):
    log_level = logging.DEBUG
    if verbosity == 0:
        log_level = logging.ERROR
    elif verbosity == 1:
        log_level = logging.INFO

    _logger.setLevel(log_level)
    tf.logging.set_verbosity(log_level)
