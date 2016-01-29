import requests
import os.path
import random
import math
import csv

from tqdm import tqdm

from collections import OrderedDict

from iris import log

"""
Work with downloading the original Iris dataset and convert the CSV into the
fields our model expects to parse.
"""

_logger = log.get_logger()

_MAIN_IRIS_DATA_URL = \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# A backup site including the iris dataset which is similar to that found on
# UCI's site, included because archive.ics.uci.edu was having connection issues
# while creating this script.
_BACKUP_IRIS_DATA_URL = \
    "http://cs.joensuu.fi/sipu/datasets/iris.data.txt"

_SEPAL_CSV_FIELDNAMES = [
    "SepalLength",
    "SepalWidth",
    "PetalLength",
    "PetalWidth",
    "Species",
]

_ONEHOT_CSV_FIELDNAMES = [
    "Species",
    "OneHot",
]


def parse_raw_iris_csv(raw_csv_filename):
    """
    A raw CSV is a file downloaded in the format expected from the UCI archive.

    Parameters
    ----------
    raw_csv_filename : str
        Relative filename of the CSV file downloaded from UCI.

    Yields
    ------
    row : dict
        Each row of the CSV with keys which match _SEPAL_CSV_FIELDNAMES.
    """
    _logger.info("Opening raw CSV file %s.", raw_csv_filename)
    with open(raw_csv_filename, "r") as raw_csv_file:
        reader = csv.DictReader(
            raw_csv_file,
            fieldnames=_SEPAL_CSV_FIELDNAMES)

        for row in reader:
            _logger.debug("Raw Row: %s", row)
            yield row


def download_iris_data(output_location, use_backup_iris_data_url):
    """
    Download the Iris dataset using the requests library and showing progress
    updates using tqdm.

    Parameters
    ----------
    output_location : str
        Directory to save the downloaded CSV into. The filename will be
        overwritten with a name of iris-data-raw.csv.
    use_backup_iris_data_url : bool
        If UCI's archive is offline, there is a backup URL which can be used to
        download the information from.

    Returns
    -------
    output_filename : str
        The filename of downloaded file or the existing file if it already
        exists.
    downloaded : bool
        If no raw CSV is found, the file will be downloaded and this will return
        True, otherwise it is False.

    Notes
    -----
    It's important to notice that if the file exists already, it won't be
    downloaded a second time.
    """
    output_filename = "{ol}/iris-data-raw.csv".format(
        ol=output_location)

    if os.path.exists(output_filename):
        _logger.info("Downloaded files already exist, skipping download.")
        return output_filename, False

    _logger.info("Downloading files to %s", output_filename)

    url = _MAIN_IRIS_DATA_URL
    if use_backup_iris_data_url:
        _logger.debug("Using backup URL.")
        url = _BACKUP_IRIS_DATA_URL

    iris_data_response = requests.get(url, stream=True)
    with open(output_filename, "wb") as iris_data_output:
        for block in tqdm(iris_data_response.iter_content()):
            iris_data_output.write(block)

    _logger.debug("Finished downloading new Iris data.")
    return output_filename, True


def generate_onehot_from_species(raw_csv_filename):
    """
    Parse the CSV file and convert each row's Species into a one-hot vector in
    order to be used by TensorFlow in our model.

    Parameters
    ----------
    raw_csv_filename : str
        Relative path to the downloaded CSV file including information on Iris'
        from Fishker's dataset.

    Returns
    -------
    unique_species : dict
        Mapping between Iris species names and their one-hot representation.
    """
    unique_species = OrderedDict()
    for row in parse_raw_iris_csv(raw_csv_filename):
        species = row["Species"]
        if species not in unique_species:
            unique_species[species] = None

    unique_species_count = len(unique_species)

    i = 0
    for species_name in unique_species.keys():
        unique_species[species_name] = \
            "%0*d" % (unique_species_count, 10 ** i)
        i += 1

    return unique_species


def write_sepal_csv(filename, rows):
    """
    Writes a CSV in a format which we specify as:
        Sepal Lengh, Sepal Width, Petal Length, Petal Width, One-Hot Species

    Parameters
    ----------
    filename : str
        Relative filename of output CSV.
    rows : dict
        Dictionary with keys matching _SEPAL_CSV_FIELDNAMES.

    Notes
    -----
    See comments on using lineterminator, this is important when using tf 0.6.
    """
    with open(filename, "w") as output_file:
        # Specifying the newline character is required with tf 0.6 otherwise
        # the CSV reader will fail to read our CSVs.
        writer = csv.DictWriter(
            output_file,
            lineterminator="\n",
            fieldnames=_SEPAL_CSV_FIELDNAMES)

        writer.writeheader()
        writer.writerows(rows)


def write_species_onehot_csv(raw_dir, species_onehot):
    """
    Writes a CSV in the format (Species, OneHot Representation) to match the
    one-hot version of the species class.

    Parameters
    ----------
    raw_dir : str
        Relative location to store CSV which will be saved with the raw UCI
        download.
    species_onehot : OrderedDict
        An ordered dict mapping a species name to its one-hot representation as
        a string.

    Returns
    -------
    filename : str
        The filename which the CSV was saved to.
    """
    filename = "{dir}/species-onehot.csv".format(dir=raw_dir)
    with open(filename, "w") as output_file:
        writer = csv.DictWriter(
            output_file,
            lineterminator="\n",
            fieldnames=_ONEHOT_CSV_FIELDNAMES)

        writer.writeheader()
        for species, onehot in species_onehot.items():
            writer.writerow({"Species": species, "OneHot": onehot})

    return filename


def read_species_onehot_csv(raw_dir):
    """
    Reads the species one-hot vector from the CSV file which was saved while
    the raw UCI data was being prepared for training.

    Parameters
    ----------
    raw_dir : str
        Directory containing the species one-hot CSV.

    Yields
    ------
    species_onehot : tuple(species, onehot)
        Each row of the CSV translated into species and one-hot.
    """
    filename = "{dir}/species-onehot.csv".format(dir=raw_dir)
    with open(filename, "r") as species_csv_file:
        reader = csv.DictReader(
            species_csv_file,
            fieldnames=_ONEHOT_CSV_FIELDNAMES)

        for row in reader:
            yield row["Species"], row["OneHot"]


def split_test_train(raw_csv_filename, species_onehot, test_dir, train_dir):
    """
    Split a CSV downloaded from UCI into a test and train dataset. We use 90% of
    the data for training and no cross validation. The reasoning is because
    there are only 150 examples in Fishker's dataset.

    Parameters
    ----------
    raw_csv_filename : str
        Downloaded CSV from UCI.
    species_onehot : dict
        Dictionary with keys of Iris species and a value of their one-hot
        representation as a string.
    test_dir : str
        Relative directory to store our test dataset.
    train_dir : str
        Relative directory to store our train dataset.
    """
    examples = []
    for row in parse_raw_iris_csv(raw_csv_filename):
        current_species = row["Species"]
        iris_with_onehot = row
        # Convert the species to be the one-hot representation.
        iris_with_onehot["Species"] = species_onehot[current_species]

        examples.append(iris_with_onehot)

    random.shuffle(examples)
    number_of_examples = len(examples)

    # We're not doing a cross validation set because the Iris dataset is so
    # small.
    # Using 90% of the examples for training and the remaining for test.
    train_count = math.ceil(number_of_examples * 0.9)

    write_sepal_csv(
        "{train_dir}/iris-data-train.csv".format(train_dir=train_dir),
        examples[0:train_count])
    write_sepal_csv(
        "{test_dir}/iris-data-test.csv".format(test_dir=test_dir),
        examples[train_count:])


def prepare(raw_dir, test_dir, train_dir, use_backup_iris_data_url=False):
    """
    Prepare our test and train dataset by downloading raw Iris data, create
    one-hot representation of Iris species and split data into a test and train
    dataset.

    Parameters
    ----------
    raw_dir : str
        Relative location to store CSV downloaded from UCI.
    test_dir : str
        Relative location to store test dataset.
    train_dir : str
        Relative location to store train dataset.
    use_backup_iris_data_url : bool, optional
        If UCI's Archive isn't responding, use a backup location which also
        hosts the same CSV.
    """
    downloaded_filename, downloaded = download_iris_data(
        raw_dir,
        use_backup_iris_data_url)

    species_onehot = generate_onehot_from_species(downloaded_filename)

    # Saving the one-hot so other processes may use it.
    write_species_onehot_csv(raw_dir, species_onehot)

    split_test_train(downloaded_filename, species_onehot, test_dir, train_dir)
