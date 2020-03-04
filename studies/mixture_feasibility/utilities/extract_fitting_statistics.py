"""A script to build the fitting options for the simulation / reweighting
comparison study.
"""
import argparse
import json
import os
from collections import defaultdict
from glob import glob

import numpy
import pandas
from evaluator import unit
from evaluator.attributes import UNDEFINED
from evaluator.client import RequestResult
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.substances import MoleFraction
from forcebalance.evaluator_io import Evaluator_SMIRNOFF
from forcebalance.nifty import lp_load

from nistdataselection.utils.utils import property_to_snake_case


def generate_statistics(target_options, target_data_set, estimated_data_set):
    """Generate a pandas data frame containing statistics about a
    particular optimization iteration.

    Parameters
    ----------
    target_options: Evaluator_SMIRNOFF.OptionsFile
        The options used during the optimization.
    target_data_set: PhysicalPropertyDataSet
        The reference data set which was optimized against.
    estimated_data_set: PhysicalPropertyDataSet
        The reference data evaluated using simulations.

    Returns
    -------
    dict of str and pandas.DataFrame
        The statistics per type of property
    """

    target_properties_by_id = {x.id: x for x in target_data_set}
    estimated_properties_by_id = {x.id: x for x in estimated_data_set}

    property_ids = [*target_properties_by_id.keys()]

    normalised_weights = {}

    for property_type in target_data_set.property_types:

        number_of_properties = len([*target_data_set.properties_by_type(property_type)])

        normalised_weights[property_type] = (
            target_options.weights[property_type] / number_of_properties
        )

    data_per_property = defaultdict(list)
    data_frames_per_property = {}

    for property_id in property_ids:

        target_property = target_properties_by_id[property_id]
        estimated_property = estimated_properties_by_id[property_id]

        temperature = target_property.thermodynamic_state.temperature.to(
            unit.kelvin
        ).magnitude
        pressure = target_property.thermodynamic_state.pressure.to(
            unit.kilopascal
        ).magnitude

        components = []
        mole_fractions = []

        for component in target_property.substance:

            for x in target_property.substance.get_amounts(component):

                if not isinstance(x, MoleFraction):
                    continue

                mole_fractions.append(x.value)
                break

            components.append(component.smiles)

        property_type = target_property.__class__.__name__
        default_unit = target_options.denominators[property_type].units

        target_value = target_property.value.to(default_unit).magnitude
        target_std = (
            numpy.nan
            if target_property.uncertainty is UNDEFINED
            else target_property.uncertainty.to(default_unit).magnitude
        )

        estimated_value = estimated_property.value.to(default_unit).magnitude
        estimated_std = (
            numpy.nan
            if estimated_property.uncertainty is UNDEFINED
            else estimated_property.uncertainty.to(default_unit).magnitude
        )

        delta = target_value - estimated_value
        delta_sqr = delta ** 2

        weight = normalised_weights[property_type]
        denominator = (
            target_options.denominators[property_type].to(default_unit).magnitude
        )

        objective_contribution = weight * (delta / denominator) ** 2

        data_row = {
            "Temperature (K)": temperature,
            "Pressure (kPa)": pressure,
            "N Components": len(target_property.substance),
        }

        for index in range(len(components)):
            data_row[f"Component {index + 1}"] = components[index]
            data_row[f"Mole Fraction {index + 1}"] = mole_fractions[index]

        data_row[f"Target {property_type} Value ({default_unit:~})"] = target_value
        data_row[f"Target {property_type} Uncertainty ({default_unit:~})"] = target_std

        data_row[
            f"Estimated {property_type} Value ({default_unit:~})"
        ] = estimated_value
        data_row[
            f"Estimated {property_type} Uncertainty ({default_unit:~})"
        ] = estimated_std

        data_row["Delta^2"] = delta_sqr
        data_row["Weight"] = weight
        data_row["Denom"] = denominator
        data_row["Term"] = objective_contribution

        data_per_property[property_type].append(data_row)

    for property_type, data_rows in data_per_property.items():
        data_frames_per_property[property_type] = pandas.DataFrame(data_rows)

    return data_frames_per_property


def main(options_path, data_set_path, output_directory):

    target_data_set = PhysicalPropertyDataSet.from_json(data_set_path)
    target_options = Evaluator_SMIRNOFF.OptionsFile.from_json(options_path)

    target_name = os.path.split(os.path.dirname(data_set_path))[-1]

    if os.path.split(os.path.dirname(options_path))[-1] != target_name:
        raise ValueError("The data set and options file belong to different targets.")

    os.makedirs(output_directory, exist_ok=True)

    # Determine how many iterations ForceBalance has completed.
    n_iterations = len(glob(f"optimize.tmp/{target_name}/iter*"))

    objective_function = []

    for iteration in range(n_iterations):

        folder_name = "iter_" + str(iteration).zfill(4)
        file_path = f"optimize.tmp/{target_name}/{folder_name}/results.json"

        iteration_results = RequestResult.from_json(file_path)
        estimated_properties = iteration_results.estimated_properties

        data_frames_per_property = generate_statistics(
            target_options, target_data_set, estimated_properties
        )

        for property_type, data_frame in data_frames_per_property.items():

            property_type = property_to_snake_case(property_type)

            output_path = os.path.join(
                output_directory, f"{folder_name}_{property_type}.csv"
            )
            data_frame.to_csv(output_path, index=False)

        # Pull out the objective function
        file_path = f"optimize.tmp/{target_name}/{folder_name}/objective.p"

        statistics = lp_load(file_path)
        objective_function.append(statistics["X"])

    # Save out the objective function
    with open(os.path.join(output_directory, "objective_function.json"), "w") as file:
        json.dump(objective_function, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extracts statistics about a force balance optimization as"
        "pandas csv files, including target and estimated values, their difference"
        "and their contributions to the objective function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--options",
        "-opt",
        type=str,
        help="The file path to the targets options file.",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        "-dat",
        type=str,
        help="The file path to the training data set.",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="The name of the directory to store the outputs in.",
        default="statistics",
        required=False,
    )

    args = parser.parse_args()
    main(args.options, args.dataset, args.output)
