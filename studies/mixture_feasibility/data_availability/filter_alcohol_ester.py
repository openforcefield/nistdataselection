"""The purpose of this script is to extract all of the density (pure and binary), hvap,
vexcess and hmix data which was measured for esters, acids and alcohols.

The found data will be stored in a `all_alcohol_ester_data` collection for use by
other scripts.
"""

import logging
import os

from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection import processing
from nistdataselection.curation.filtering import filter_by_checkmol, filter_by_elements
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import data_frame_to_pdf, property_to_snake_case

logger = logging.getLogger(__name__)

chemical_environment_codes = {
    "hydroxy": "027",
    "alcohol": "028",
    "caboxylic_acid": "076",
    "ester": "078",
}


def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    root_data_directory = "../../shared/filtered_data"

    # Define those properties / environments of interest.
    properties_of_interest = [
        (Density, SubstanceType.Pure),
        (Density, SubstanceType.Binary),
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (ExcessMolarVolume, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
    ]

    chemical_environments = {
        SubstanceType.Pure: [[*chemical_environment_codes.values()]],
        SubstanceType.Binary: [
            [
                chemical_environment_codes["hydroxy"],
                chemical_environment_codes["alcohol"],
            ],
            [
                chemical_environment_codes["caboxylic_acid"],
                chemical_environment_codes["ester"],
            ],
        ],
    }

    # Make directories to store the data in.
    output_directory = "all_alcohol_ester_data"
    os.makedirs(output_directory, exist_ok=True)

    for property_tuple in properties_of_interest:

        property_type, substance_type = property_tuple

        data_set = processing.load_processed_data_set(
            root_data_directory, property_type, substance_type
        )

        # Start by filtering out any substances not composed of O, C, H
        data_set = filter_by_elements(data_set, "C", "H", "O")

        # Next filter out any substances which aren't alcohols, esters or acids.
        data_set = filter_by_checkmol(data_set, *chemical_environments[substance_type])

        # Save the filtered data set.
        processing.save_processed_data_set(
            output_directory, data_set, property_type, substance_type
        )

        # Save out a pdf of all smiles patterns (/ tuples of smiles patterns).
        property_type = property_to_snake_case(property_type)

        file_name = f"{property_type}_{str(substance_type.value)}.pdf"
        file_path = os.path.join(output_directory, file_name)

        data_frame_to_pdf(data_set, file_path)


if __name__ == "__main__":
    main()
