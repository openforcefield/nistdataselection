"""A script to filter out all of the training set
compounds from the available data.
"""
import os
from tempfile import TemporaryDirectory

import pandas
from evaluator import unit
from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection.curation.filtering import filter_by_smiles
from nistdataselection.curation.selection import StatePoint, select_data_points
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.pandas import data_frame_to_smiles_tuples
from nistdataselection.utils.utils import (
    data_frame_to_pdf,
    property_to_snake_case,
    substance_type_to_int,
)


def main():

    output_directory = "test_sets"
    os.makedirs(output_directory, exist_ok=True)

    # Define the types of property which are of interest.
    data_of_interest = {
        (EnthalpyOfMixing, SubstanceType.Binary): {
            "alcohol_alcohol": ["one_in_training", "not_in_training"],
            "alcohol_ester": ["not_in_training"],
            "ester_ester": ["both_in_training", "not_in_training"],
        },
        (ExcessMolarVolume, SubstanceType.Binary): {
            "alcohol_alcohol": ["both_in_training", "not_in_training"],
            "alcohol_ester": ["not_in_training"],
            "ester_ester": ["both_in_training", "not_in_training"],
        },
        (Density, SubstanceType.Binary): {
            "alcohol_alcohol": ["both_in_training", "not_in_training"],
            "alcohol_ester": ["not_in_training"],
            "ester_ester": ["both_in_training", "not_in_training"],
        },
        (Density, SubstanceType.Pure): {
            "alcohol_alcohol": ["not_in_training"],
            "alcohol_ester": ["not_in_training"],
            "ester_ester": ["not_in_training"],
        },
        (EnthalpyOfVaporization, SubstanceType.Pure): {
            "alcohol_alcohol": ["not_in_training"],
            "alcohol_ester": ["not_in_training"],
            "ester_ester": ["not_in_training"],
        },
    }

    properties_of_interest = [*data_of_interest]

    # Define the state points of interest
    target_states = {
        SubstanceType.Pure: [
            StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (1.0,)),
        ],
        SubstanceType.Binary: [
            StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.25, 0.75)),
            StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.50, 0.50)),
            StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.75, 0.25)),
        ],
    }

    target_states = {(x, y): target_states[y] for x, y in properties_of_interest}

    # Load in all of the available data which is of interest.
    data_frames_by_property_type = {}

    for property_type in data_of_interest:

        data_frames = []

        for environment_type in data_of_interest[property_type]:

            for partition_of_interest in data_of_interest[property_type][environment_type]:

                data_directory = os.path.join(
                    "partitioned_data", environment_type, partition_of_interest
                )
                data_frames.append(
                    load_processed_data_set(data_directory, *property_type)
                )

        data_frames_by_property_type[property_type] = pandas.concat(
            data_frames, ignore_index=True, sort=False
        )

    # Apply a rough filter to cut down on the amount of pure density data
    smiles_to_include = []

    for property_tuple, data_frame in data_frames_by_property_type.items():

        property_type, substance_type = property_tuple

        if property_type == Density and substance_type == SubstanceType.Pure:
            continue

        substance_smiles = data_frame_to_smiles_tuples(data_frame)
        smiles_to_include.extend(x for y in substance_smiles for x in y)

    smiles_to_include = set(smiles_to_include)

    data_frames_by_property_type[(Density, SubstanceType.Pure)] = filter_by_smiles(
        data_frames_by_property_type[(Density, SubstanceType.Pure)],
        smiles_to_include,
        None
    )

    # Select the data points to include.
    with TemporaryDirectory() as working_directory:

        for property_type, data_frame in data_frames_by_property_type.items():
            save_processed_data_set(working_directory, data_frame, *property_type)

        full_data_set = select_data_points(
            data_directory=working_directory,
            chosen_substances=None,
            target_state_points=target_states,
        )

    full_data_frame = full_data_set.to_pandas()
    full_data_frame.to_csv(os.path.join(output_directory, "full_set.csv"), index=False)

    # Save out the data set
    full_data_set.json(os.path.join(output_directory, "full_set.json"))

    # Create views of the data in the data set.
    for property_type, substance_type in properties_of_interest:

        n_components = substance_type_to_int[substance_type]

        # Filter by number of components
        data_frame = full_data_frame[full_data_frame["N Components"] == n_components]
        data_frame = data_frame.dropna(axis=1, how="all")

        property_header = (
            f"{property_type.__name__} Value ({property_type.default_unit():~})"
        )
        data_frame = data_frame.dropna(axis=0, how="all", subset=[property_header])

        property_name = property_to_snake_case(property_type)

        file_name = f"{property_name}_{str(substance_type.value)}"
        file_path = os.path.join(output_directory, file_name)

        data_frame.to_csv(f"{file_path}.csv", index=False)
        data_frame_to_pdf(data_frame, f"{file_path}.pdf")


if __name__ == "__main__":
    main()
