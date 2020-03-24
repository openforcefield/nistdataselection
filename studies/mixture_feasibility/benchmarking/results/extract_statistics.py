"""A script to build the fitting options for the simulation / reweighting
comparison study.
"""
import itertools
import logging
import os
from collections import defaultdict

import numpy
import pandas
from evaluator import unit
from evaluator.attributes import UNDEFINED
from evaluator.client import RequestResult
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.substances import MoleFraction

from nistdataselection.curation.filtering import filter_by_checkmol
from nistdataselection.utils.pandas import data_frame_to_smiles_tuples
from nistdataselection.utils.utils import (
    SubstanceType,
    chemical_environment_codes,
    int_to_substance_type,
    property_to_file_name,
)

logger = logging.getLogger(__name__)


def find_training_smiles():
    """Returns the smiles of all of the substances which
    appeared in the training set.

    Returns
    -------
    list of tuple of str
        The smiles patterns of the training substances.
    """

    # Find those alcohols which were included in the training set
    training_set = PhysicalPropertyDataSet.from_json(
        os.path.join(
            "..",
            "..",
            "pure_mixture_optimisation",
            "force_balance",
            "h_mix_rho_x_rho_pure_h_vap",
            "targets",
            "mixture_data",
            "training_set.json",
        )
    ).to_pandas()

    training_smiles = data_frame_to_smiles_tuples(training_set)
    training_smiles = set(x for y in training_smiles for x in y)

    return training_smiles


def generate_statistics(experimental_data_set, estimated_data_set):
    """Generate a pandas data frame containing the values of
    both an experimentally measured data set, and an estimated
    data set.

    Parameters
    ----------
    experimental_data_set: PhysicalPropertyDataSet
        The experimental data set.
    estimated_data_set: PhysicalPropertyDataSet
        The estimated data set.

    Returns
    -------
    dict of tuple of type of PhysicalProperty and SubstanceType and pandas.DataFrame
        The statistics per type of property
    """

    experimental_properties_by_id = {x.id: x for x in experimental_data_set}
    estimated_properties_by_id = {x.id: x for x in estimated_data_set}

    property_ids = [*experimental_properties_by_id.keys()]

    data_per_property = defaultdict(list)
    data_frames_per_property = {}

    for property_id in property_ids:

        experimental_property = experimental_properties_by_id[property_id]

        if property_id not in estimated_properties_by_id:

            logger.info(f"{property_id} was missing from the estimated set.")
            continue

        estimated_property = estimated_properties_by_id[property_id]

        temperature = experimental_property.thermodynamic_state.temperature.to(
            unit.kelvin
        ).magnitude
        pressure = experimental_property.thermodynamic_state.pressure.to(
            unit.kilopascal
        ).magnitude

        components = []
        mole_fractions = []

        for component in experimental_property.substance:

            for x in experimental_property.substance.get_amounts(component):

                if not isinstance(x, MoleFraction):
                    continue

                mole_fractions.append(x.value)
                break

            components.append(component.smiles)

        property_class = experimental_property.__class__
        property_type = property_class.__name__

        default_unit = property_class.default_unit()

        experimental_value = experimental_property.value.to(default_unit).magnitude
        experimental_std = (
            numpy.nan
            if experimental_property.uncertainty is UNDEFINED
            else experimental_property.uncertainty.to(default_unit).magnitude
        )

        estimated_value = estimated_property.value.to(default_unit).magnitude
        estimated_std = (
            numpy.nan
            if estimated_property.uncertainty is UNDEFINED
            else estimated_property.uncertainty.to(default_unit).magnitude
        )

        data_row = {
            "Temperature (K)": temperature,
            "Pressure (kPa)": pressure,
            "N Components": len(experimental_property.substance),
        }

        for index in range(len(components)):
            data_row[f"Component {index + 1}"] = components[index]
            data_row[f"Mole Fraction {index + 1}"] = mole_fractions[index]

        data_row[
            f"Target {property_type} Value ({default_unit:~})"
        ] = experimental_value
        data_row[
            f"Target {property_type} Uncertainty ({default_unit:~})"
        ] = experimental_std

        data_row[
            f"Estimated {property_type} Value ({default_unit:~})"
        ] = estimated_value
        data_row[
            f"Estimated {property_type} Uncertainty ({default_unit:~})"
        ] = estimated_std

        data_per_property[property_class].append(data_row)

    for property_type, data_rows in data_per_property.items():

        data_frame = pandas.DataFrame(data_rows)

        n_components = data_frame["N Components"].max()

        for n in range(n_components):

            substance_type = int_to_substance_type[n + 1]

            component_data_frame = data_frame[data_frame["N Components"] == (n + 1)]
            component_data_frame = component_data_frame.dropna(axis=1, how="all")

            data_frames_per_property[
                (property_type, substance_type)
            ] = component_data_frame

    return data_frames_per_property


def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the experimental test set.
    test_data_set = PhysicalPropertyDataSet.from_json("test_data_set.json")

    study_names = [
        "openff-1.0.0",
        "h_mix_rho_x",
        "h_mix_rho_x_rho_pure",
        "h_mix_rho_x_rho_pure_h_vap",
        "h_mix_v_excess",
        "rho_pure_h_vap",
    ]

    environments_of_interest = {
        "alcohol": [
            chemical_environment_codes["hydroxy"],
            chemical_environment_codes["alcohol"],
        ],
        "ester": [
            chemical_environment_codes["caboxylic_acid"],
            chemical_environment_codes["ester"],
        ],
    }

    environment_pairs = [(x, x) for x in environments_of_interest]
    # noinspection PyTypeChecker
    environment_pairs.extend(itertools.combinations(environments_of_interest, 2))

    for study_name in study_names:

        logger.info(f"Processing {study_name}")

        output_directory = os.path.join("all_results", study_name)
        os.makedirs(output_directory, exist_ok=True)

        # Extract the matching experimental and estimated statistics.
        benchmark_results = RequestResult.from_json(
            os.path.join("raw_results", f"{study_name}.json")
        )
        estimated_properties = benchmark_results.estimated_properties

        data_frames_per_property = generate_statistics(
            test_data_set, estimated_properties
        )

        # Save the data frames to disk.
        for property_tuple, data_frame in data_frames_per_property.items():

            file_name = property_to_file_name(*property_tuple)
            output_path = os.path.join(output_directory, f"{file_name}.csv")

            data_frame.to_csv(output_path, index=False)

        # Split the full results into the mixture type, and whether the mixture
        # components appears in the training set
        training_smiles = find_training_smiles()

        output_directory = os.path.join("partitioned_results", study_name)

        for property_tuple, data_frame in data_frames_per_property.items():

            property_type, substance_type = property_tuple

            if substance_type != SubstanceType.Binary:
                continue

            file_name = property_to_file_name(*property_tuple)

            for environment_types in environment_pairs:

                environment_type_1, environment_type_2 = environment_types

                # Apply the filters to the pure properties.
                chemical_environments = [
                    environments_of_interest[environment_type_1],
                    environments_of_interest[environment_type_2],
                ]

                environment_data_frame = filter_by_checkmol(
                    data_frame, *chemical_environments
                )

                # Extract properties where neither component appears in
                # in the training set.
                by_type_data_frame = environment_data_frame[
                    ~environment_data_frame["Component 1"].isin(training_smiles)
                    & ~environment_data_frame["Component 2"].isin(training_smiles)
                ]

                base_directory = os.path.join(
                    output_directory,
                    f"{environment_type_1}_{environment_type_2}",
                    "not_in_training",
                )
                os.makedirs(base_directory, exist_ok=True)

                by_type_data_frame.to_csv(
                    os.path.join(base_directory, file_name + ".csv"), index=False
                )

                # Extract properties where both components appear in
                # in the training set.
                by_type_data_frame = environment_data_frame[
                    environment_data_frame["Component 1"].isin(training_smiles)
                    & environment_data_frame["Component 2"].isin(training_smiles)
                ]

                base_directory = os.path.join(
                    output_directory,
                    f"{environment_type_1}_{environment_type_2}",
                    "both_in_training",
                )
                os.makedirs(base_directory, exist_ok=True)

                by_type_data_frame.to_csv(
                    os.path.join(base_directory, file_name + ".csv"), index=False
                )

                # Extract properties where only one component appears in
                # in the training set.
                by_type_data_frame = environment_data_frame[
                    (
                        environment_data_frame["Component 1"].isin(training_smiles)
                        & ~environment_data_frame["Component 2"].isin(training_smiles)
                    )
                    | (
                        ~environment_data_frame["Component 1"].isin(training_smiles)
                        & environment_data_frame["Component 2"].isin(training_smiles)
                    )
                ]

                base_directory = os.path.join(
                    output_directory,
                    f"{environment_type_1}_{environment_type_2}",
                    "one_in_training",
                )
                os.makedirs(base_directory, exist_ok=True)

                by_type_data_frame.to_csv(
                    os.path.join(base_directory, file_name + ".csv"), index=False
                )


if __name__ == "__main__":
    main()
