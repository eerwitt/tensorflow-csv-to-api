import tensorflow as tf

import logging

"""
Base logging support which also edits the level of logging used in TensorFlow.
"""

# Default logger for keeping track of steps.
_logger = logging.getLogger("iris")

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
_logger.addHandler(_handler)


def get_logger():
    """
    Allow access to the global logger used in the application.
    """
    return _logger


def set_verbosity(verbosity):
    """
    Adjust this logger's verbosity level on a scale which is:
        0 => Error only logging
        1 => Info logging
        Anything else => Debug logging

    Parameters
    ----------
    verbosity : int
        Level of messages to be reported.
    """
    log_level = logging.DEBUG
    if verbosity == 0:
        log_level = logging.ERROR
    elif verbosity == 1:
        log_level = logging.INFO

    _logger.setLevel(log_level)
    tf.logging.set_verbosity(log_level)
