"""This script finds those substances for which their is data
of multiple specific types available (i.e those substances which
have both binary mass density and enthalpy of mixing data available).

All data for such substances is then extracted and stored in the
`data_by_environments/{environment_1}_{environment_2}/common_data`
directory.
"""
import logging
import os
from glob import glob

from evaluator.properties import Density, EnthalpyOfMixing, ExcessMolarVolume

from nistdataselection.curation.filtering import filter_by_substance_composition
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils.pandas import data_frame_to_smiles_tuples
from nistdataselection.utils.utils import SubstanceType, smiles_to_pdf


def main():

    root_output_directory = "data_by_environments"

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Define the types of data to find.
    properties_of_interest = [
        [(EnthalpyOfMixing, SubstanceType.Binary), (Density, SubstanceType.Binary)],
        [
            (EnthalpyOfMixing, SubstanceType.Binary),
            (ExcessMolarVolume, SubstanceType.Binary),
        ],
        [
            (EnthalpyOfMixing, SubstanceType.Binary),
            (Density, SubstanceType.Binary),
            (ExcessMolarVolume, SubstanceType.Binary),
        ],
    ]

    # Define some shorter file names to use:
    type_to_file_name = {
        (Density, SubstanceType.Binary): "rho_x",
        (EnthalpyOfMixing, SubstanceType.Binary): "h_mix",
        (ExcessMolarVolume, SubstanceType.Binary): "v_excess",
    }

    # Define which types of mixtures we are interested in, e.g.
    # alcohol-alcohol, alcohol-ester etc.
    environments_of_interest = [
        os.path.basename(x) for x in glob("data_by_environments/*")
    ]

    for environment_of_interest in environments_of_interest:

        data_directory = os.path.join(
            "data_by_environments", environment_of_interest, "all_data"
        )

        os.makedirs(
            os.path.join(root_output_directory, environment_of_interest, "common_data"),
            exist_ok=True,
        )

        for property_type_set in properties_of_interest:

            # Find the set of substances which are common to all of the
            # specified property types.
            all_substance_smiles = []

            for property_type, substance_type in property_type_set:

                data_frame = load_processed_data_set(
                    data_directory, property_type, substance_type
                )

                substance_smiles = set(data_frame_to_smiles_tuples(data_frame))
                all_substance_smiles.append(substance_smiles)

            common_substance_smiles = set.intersection(*all_substance_smiles)

            # Save the common substances to a pdf file.
            file_name = "_".join(type_to_file_name[x] for x in property_type_set)

            file_path = os.path.join(
                root_output_directory,
                environment_of_interest,
                "common_data",
                f"{file_name}.pdf",
            )

            if len(common_substance_smiles) > 0:
                smiles_to_pdf(list(common_substance_smiles), file_path)

            # Output the common data to the `common_data` directory.
            output_directory = os.path.join(
                root_output_directory, environment_of_interest, "common_data", file_name
            )

            for property_type, substance_type in property_type_set:

                data_frame = load_processed_data_set(
                    data_directory, property_type, substance_type
                )

                data_frame = filter_by_substance_composition(
                    data_frame, common_substance_smiles, None
                )

                save_processed_data_set(
                    output_directory, data_frame, property_type, substance_type
                )


if __name__ == "__main__":
    main()
