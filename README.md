# Tensorflow: From CSV to API

Code from [this tutorial](https://eerwitt.github.io/2016/01/14/tensorflow-from-csv-to-api/) about creating a REST API which uses a model trained using TensorFlow. The code goes through the entire process from the downloading of a CSV training file to hosting the model in an API.

## Installation

Using Python 3 in a [virtual environment](https://virtualenv.readthedocs.org/en/latest/), install the required packages via [pip](https://pip.pypa.io/en/stable/).

```zsh
pip install -r requirements.txt
```

You might need to edit the version of TensorFlow in the requirements file. Details are found on [TensorFlow's installation guide](https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html).

## Running

There are four commands which include a number of arguments.

| Command | Explanation |
| :--- | :--- |
| `python iris-network-download.py -vv` | Download example Iris data from UCI and convert the CSV into a test and train dataset with the Iris species converted to a one-hot vector representation. |
| `python iris-network-train.py -vv` | Train a Neural Network using feed forward learning and save checkpoint models to a directory. |
| `python iris-network-test.py -vv` | Test the trained Neural Network to check for accuracy. |
| `python iris-network-predict.py --feature 5.5,4.2,1.4,0.2 -vv` | Check the prediction for an Iris described by the features `5.5,4.2,1.4,0.2` which correspond to Sepal Length, Sepal Width, Petal Length and Petal Width. |
| `gunicorn web.iris:api` | Start a `gunicorn` webserver to host the Falcon API. |

## Testing

This code is not meant to be used in production and doesn't provide necessary tests to validate its functionality.

If you're writing TensorFlow tests there [is a useful Python class](https://github.com/tensorflow/tensorflow/blob/f13d006e097c4e8010a4ad3ad2018a0369f5dc19/tensorflow/python/framework/test_util.py) which makes testing a graph fairly trivial.
