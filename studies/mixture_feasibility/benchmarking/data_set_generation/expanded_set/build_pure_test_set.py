"""A script to select the pure property data points to include
in the expanded test set.
"""
import logging
import os
from tempfile import TemporaryDirectory

import pandas
from evaluator import unit
from evaluator.properties import Density, EnthalpyOfVaporization

from nistdataselection.curation.filtering import (
    filter_by_elements,
    filter_by_smiles,
    filter_by_smirks,
    filter_by_temperature,
    filter_undefined_stereochemistry,
)
from nistdataselection.curation.selection import StatePoint, select_data_points
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils.utils import SubstanceType, data_frame_to_pdf

logger = logging.getLogger(__name__)


def load_training_components():
    """Loads in the pure training molecules. These will
    be used to exclude training compounds from the test
    set.

    Returns
    -------
    set of str
        The substances in the training set.
    """

    training_set = pandas.read_csv(
        os.path.join(
            "..",
            "..",
            "..",
            "pure_optimisation",
            "data_set_generation",
            "expanded_set",
            "training_set.csv",
        )
    )

    components = {x["Component 1"] for _, x in training_set.iterrows()}
    return components


def filter_data(data_frame):
    """Filters out data which is not of interest for the test
    set.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame to filter.

    Returns
    -------
    pandas.DataFrame
        The filtered data frame.
    """

    # Filter to be closer to ambient.
    data_frame = filter_by_temperature(
        data_frame, 280.0 * unit.kelvin, 315.0 * unit.kelvin
    )

    # Filter out long chain molecules, 3 + 4 membered rings
    # and 1, 3 carbonyl compounds where one of the carbonyls
    # is a ketone (cases where the enol form may be present in
    # non-negligible amounts).
    data_frame = filter_by_smirks(
        data_frame,
        None,
        [
            # 3 + 4 membered rings.
            "[#6r3]",
            "[#6r4]",
            # Long chain alkane /ether
            "[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]",
            # 1, 3 carbonyls with at least one ketone carbonyl.
            "[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](=[#8])-[#6]",
        ],
    )

    allowed_elements = ["H", "C", "O"]
    data_frame = filter_by_elements(data_frame, *allowed_elements)

    # Filter out any molecules with undefined stereochemistry
    data_frame = filter_undefined_stereochemistry(data_frame)

    return data_frame


def main():

    logging.basicConfig(level=logging.INFO)

    root_output_directory = "test_sets"
    os.makedirs(root_output_directory, exist_ok=True)

    # Define the types of property which are of interest.
    properties_of_interest = [
        (Density, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure),
    ]

    # Define the state we would ideally chose data points at.
    target_states = [
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (1.0,)),
    ]
    target_states = {x: target_states for x in properties_of_interest}

    # Define the environments of interest.
    environments_of_interest = ["alcohol", "ester", "alkane", "ether", "ketone"]

    # Load in the training substances so we can avoid selecting
    # them for the test set.
    training_smiles = load_training_components()

    with TemporaryDirectory() as data_directory:

        # Apply the filters to the available data.
        for property_of_interest in properties_of_interest:

            data_frames = []

            for environment in environments_of_interest:

                data_frame = load_processed_data_set(
                    os.path.join(
                        "..",
                        "..",
                        "..",
                        "data_availability",
                        "data_by_environments",
                        f"{environment}_{environment}",
                        "all_data",
                    ),
                    *property_of_interest,
                )

                data_frames.append(data_frame)

            data_frame = pandas.concat(data_frames, ignore_index=True, sort=False)

            data_frame = filter_data(data_frame)
            data_frame = filter_by_smiles(data_frame, training_smiles, None)

            save_processed_data_set(data_directory, data_frame, *property_of_interest)

        # Determine which components have enthalpy of vaporization
        # measurements. These will be the compounds which will be
        # included in the pure test set.
        h_vap_data_frame = load_processed_data_set(
            data_directory, EnthalpyOfVaporization, SubstanceType.Pure
        )

        test_set_components = {*h_vap_data_frame["Component 1"]}
        test_set_components = [(x,) for x in test_set_components]

        # Select the data points.
        selected_data_set = select_data_points(
            data_directory=data_directory,
            chosen_substances=test_set_components,
            target_state_points=target_states,
        )

    selected_data_set.json(os.path.join(root_output_directory, "pure_set.json"))

    selected_data_frame = selected_data_set.to_pandas()
    selected_data_frame.to_csv(
        os.path.join(root_output_directory, "pure_set.csv"), index=False
    )

    data_frame_to_pdf(
        selected_data_frame, os.path.join(root_output_directory, "pure_set.pdf")
    )


if __name__ == "__main__":
    main()
