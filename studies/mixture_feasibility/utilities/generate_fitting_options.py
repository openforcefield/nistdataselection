"""A script to build the fitting options for the simulation / reweighting
comparison study.
"""
import argparse
import copy

import numpy
from evaluator.client import ConnectionOptions, RequestOptions
from evaluator.datasets import PhysicalPropertyDataSet
from forcebalance.evaluator_io import Evaluator_SMIRNOFF


def calculate_denominators(data_set):
    """Choose the ForceBalance denominators so that the data range is effectively
    between 0.0 and 1.0.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to normalize.

    Returns
    -------
    dict of str and pint.Quantity
        The calculated denominators.
    """
    denominators = {}

    for property_type in data_set.property_types:

        property_class = next(iter(data_set.properties_by_type(property_type)))

        property_data_set = copy.deepcopy(data_set)
        property_data_set.filter_by_property_types(property_type)

        default_unit = property_class.default_unit()
        values = [x.value.to(default_unit) for x in property_data_set]

        max_value = max(values)
        min_value = min(values)

        denominator_sqr = (max_value - min_value).magnitude
        denominator = numpy.sqrt(denominator_sqr) * default_unit

        denominators[property_type] = denominator

    return denominators


def main(input_data_set_path, server_port):

    # Create the options which propertyestimator should use.
    estimator_options = RequestOptions()

    # Choose which calculation layers to make available.
    estimator_options.calculation_layers = ["SimulationLayer"]

    # Load in the training data set and create schemas for each of the types
    # of property to be calculated.
    training_set = PhysicalPropertyDataSet.from_json(input_data_set_path)

    # Zero out any undefined uncertainties due to a bug in ForceBalance.
    for physical_property in training_set:
        physical_property.uncertainty = 0.0 * physical_property.default_unit()

    data_set_path = "training_set.json"
    training_set.json(data_set_path, format=True)

    # Create the force balance options
    target_options = Evaluator_SMIRNOFF.OptionsFile()
    target_options.connection_options = ConnectionOptions(
        server_address="localhost", server_port=server_port
    )
    target_options.estimation_options = estimator_options

    target_options.data_set_path = data_set_path

    # Set the property weights and denominators.
    target_options.weights = {x: 1.0 for x in training_set.property_types}
    target_options.denominators = calculate_denominators(training_set)

    # Save the options to file.
    with open("options.json", "w") as file:
        file.write(target_options.to_json())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generates a default set of ForceBalance fitting options."
        "This script will create an `options.json` file containg the fit options, "
        "and a `training_set.json` file which contains the training data set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="The port that the server will be listening on.",
        required=False,
        default=8000,
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="The file path to the training data set.",
        required=False,
    )

    args = parser.parse_args()
    main(args.file, args.port)
