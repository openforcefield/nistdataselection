"""The purpose of this script is to extract all of the density (pure and binary),
hvap, vexcess and hmix data which was measured for systems which contain specific
chemical environments.

The extracted mixture data will be partitioned by the environments they exercise
and stored in the `data_by_environments` directory.
"""
import functools
import itertools
import logging
import os
from multiprocessing.pool import Pool
from tempfile import TemporaryDirectory

import tqdm
from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection import processing
from nistdataselection.curation.filtering import filter_by_checkmol, filter_by_elements
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import (
    chemical_environment_codes,
    data_frame_to_pdf,
    property_to_snake_case,
)

logger = logging.getLogger(__name__)


def filter_data(
    data_directory, properties_of_interest, chemical_environments, output_directory
):
    """Filters out any measurements which where made for components which
    do not contain the chemical environments of interest.

    Parameters
    ----------
    data_directory: str
        The directory containing the unfiltered data.
    properties_of_interest: list of tuple of PropertyType and SubstanceType
        The types of properties to extract data for.
    chemical_environments: list of list of str
        A list of those chemical environments to filter by. Each list in the
        full list corresponds to the chemical environments which should be
        matched by one of the components in the system.
    output_directory: str
        The directory to store the extracted data in.
    """

    for property_tuple in properties_of_interest:

        property_type, substance_type = property_tuple

        data_set = processing.load_processed_data_set(
            data_directory, property_type, substance_type
        )

        # Start by filtering out any substances not composed of O, C, H, N, F, Cl, Br, S
        data_set = filter_by_elements(
            data_set, "C", "H", "O", "N", "F", "Cl", "Br", "S"
        )

        # Next filter out any substances which aren't alcohols, esters or acids.
        data_set = filter_by_checkmol(data_set, *chemical_environments)

        # Save the filtered data set.
        processing.save_processed_data_set(
            output_directory, data_set, property_type, substance_type
        )

        # Save out a pdf of all smiles patterns (/ tuples of smiles patterns).
        property_type = property_to_snake_case(property_type)

        file_name = f"{property_type}_{str(substance_type.value)}.pdf"
        file_path = os.path.join(output_directory, file_name)

        data_frame_to_pdf(data_set, file_path)


def apply_filters(
    environment_types,
    data_directory,
    environments_of_interest,
    mixture_properties_of_interest,
    pure_properties_of_interest,
    root_output_directory,
):

    environment_type_1, environment_type_2 = environment_types

    output_directory = os.path.join(
        root_output_directory,
        f"{environment_type_1}_{environment_type_2}",
        f"all_data",
    )

    if os.path.isdir(output_directory):
        return

    os.makedirs(output_directory, exist_ok=True)
    # Apply the filters to the pure properties.
    chemical_environments = [
        {
            *environments_of_interest[environment_type_1],
            *environments_of_interest[environment_type_2],
        }
    ]
    filter_data(
        data_directory,
        pure_properties_of_interest,
        chemical_environments,
        output_directory,
    )
    # Apply the filters to mixture properties.
    chemical_environments = [
        environments_of_interest[environment_type_1],
        environments_of_interest[environment_type_2],
    ]
    filter_data(
        data_directory,
        mixture_properties_of_interest,
        chemical_environments,
        output_directory,
    )


def main():

    root_output_directory = "data_by_environments"

    # Define the number of processes to parallelize over.
    n_processes = 20

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Define the properties and environments we are interested in.
    pure_properties_of_interest = [
        (Density, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure),
    ]
    mixture_properties_of_interest = [
        (Density, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
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
        "ether": [chemical_environment_codes["ether"]],
        "aldehyde": [chemical_environment_codes["aldehyde"]],
        "ketone": [chemical_environment_codes["ketone"]],
        "thiocarbonyl": [chemical_environment_codes["thiocarbonyl"]],
        "phenol": [chemical_environment_codes["phenol"]],
        "amine": [chemical_environment_codes["amine"]],
        "halogenated": [chemical_environment_codes["halogenated"]],
        "amide": [chemical_environment_codes["amide"]],
        "nitro": [chemical_environment_codes["nitro"]],
        "aromatic": [chemical_environment_codes["aromatic"]],
        "heterocycle": [chemical_environment_codes["heterocycle"]],
        "alkane": [""],
        "alkene": [chemical_environment_codes["alkene"]],
    }

    properties_of_interest = [
        *pure_properties_of_interest,
        *mixture_properties_of_interest,
    ]

    with TemporaryDirectory() as data_directory:

        root_data_directory = os.path.join("..", "..", "shared", "filtered_data")

        # Create a temporary directory which contains both the converted
        # mass density / excess molar volume data, and the other data of
        # interest
        for property_type, substance_type in properties_of_interest:

            if (
                property_type in [Density, ExcessMolarVolume]
                and substance_type == SubstanceType.Binary
            ):
                # Source any binary mass density or excess molar
                # volume from the full set of converted density
                # data.
                data_set = load_processed_data_set(
                    "converted_density_data", property_type, substance_type
                )

            elif property_type == EnthalpyOfVaporization:

                data_set = load_processed_data_set(
                    "sourced_h_vap_data", property_type, substance_type
                )

            else:

                data_set = load_processed_data_set(
                    root_data_directory, property_type, substance_type
                )

            save_processed_data_set(
                data_directory, data_set, property_type, substance_type
            )

        # Determine all combinations of the environments of interest.
        environment_pairs = [(x, x) for x in environments_of_interest]
        environment_pairs.extend(itertools.combinations(environments_of_interest, 2))

        with Pool(n_processes) as pool:

            x = list(
                tqdm.tqdm(
                    pool.imap(
                        functools.partial(
                            apply_filters,
                            data_directory=data_directory,
                            environments_of_interest=environments_of_interest,
                            mixture_properties_of_interest=mixture_properties_of_interest,
                            pure_properties_of_interest=pure_properties_of_interest,
                            root_output_directory=root_output_directory,
                        ),
                        environment_pairs,
                    ),
                    total=len(environment_pairs),
                )
            )

        assert x is not None


if __name__ == "__main__":
    main()
