import random
import math
import csv

from collections import OrderedDict

from network import iris


CSV_FIELDNAMES = [
    "SepalLength",
    "SepalWidth",
    "PetalLength",
    "PetalWidth",
    "Species",
]


def parse_raw_iris_csv(raw_csv_filename):
    with open(raw_csv_filename, "r") as raw_csv_file:
        reader = csv.DictReader(
            raw_csv_file,
            fieldnames=CSV_FIELDNAMES)

        for row in reader:
            yield row


def generate_onehot_from_classes(raw_csv_filename):
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


def split_test_train(raw_csv_filename, class_onehot):
    examples = []
    for row in parse_raw_iris_csv(raw_csv_filename):
        current_species = row["Species"]
        iris = row
        # Convert the species to be the onehot representation.
        iris["Species"] = class_onehot[current_species]

        examples.append(iris)

    random.shuffle(examples)
    number_of_examples = len(examples)

    # We're not doing a cross validation set because the Iris dataset is so
    # small.
    # Using 80% of the examples for training and the remaining for test.
    train_count = math.ceil(number_of_examples * 0.8)

    with open("./train/iris-data-shuffled.csv", "w") as train_file:
        train_writer = csv.DictWriter(
            train_file,
            lineterminator="\n",  # Required otherwise tf won't like newlines
            fieldnames=CSV_FIELDNAMES)

        train_writer.writeheader()
        train_writer.writerows(examples[0:train_count])

    with open("./test/iris-data-test.csv", "w") as test_file:
        test_writer = csv.DictWriter(
            test_file,
            lineterminator="\n",  # Required otherwise tf won't like newlines
            fieldnames=CSV_FIELDNAMES)

        test_writer.writeheader()
        test_writer.writerows(examples[train_count:])


if __name__ == "__main__":
    iris.download_iris_data("./raw")
    class_onehot = generate_onehot_from_classes("./raw/iris-data-raw.csv")

    split_test_train("./raw/iris-data-raw.csv", class_onehot)
