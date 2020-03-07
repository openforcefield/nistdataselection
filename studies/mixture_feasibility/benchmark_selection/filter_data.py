"""A script which will filter the available mixture data
so that

* no training mixtures appear in the test sets.
* only data measured at conditions close to ambient is retained
* longer chain molecules / aromatics are removed.

Currently we filter out some of the longer molecules, aromatics,
and molecules containing moieties not trained against to allow
the construction of a smaller, more focused initial benchmark
set.
"""
import os

from evaluator import unit
from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection.curation.filtering import (
    filter_by_smirks,
    filter_by_temperature,
    filter_undefined_stereochemistry,
)
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import data_frame_to_pdf, property_to_snake_case


def filter_data(data_directory, property_type, substance_type, output_directory):

    # Load in the data set
    data_frame = load_processed_data_set(data_directory, property_type, substance_type)

    # Filter to be close to ambient.
    data_frame = filter_by_temperature(
        data_frame, 290.0 * unit.kelvin, 305 * unit.kelvin
    )

    # Filter out aromatics, long chain molecules (>= hept), alkenes,
    # ethers, 3 + 4 membered rings
    data_frame = filter_by_smirks(
        data_frame,
        None,
        [
            "[#6a]",
            "[#6r3]",
            "[#6r4]",
            "[#6]=[#6]",
            "[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]",
            "[#6H2]-[#8X2]-[#6H2]",
        ],
    )

    # Filter out any molecules with undefined stereochemistry
    data_frame = filter_undefined_stereochemistry(data_frame)

    # Save the filtered set.
    save_processed_data_set(
        output_directory, data_frame, property_type, substance_type,
    )

    property_type = property_to_snake_case(property_type)
    file_name = f"{property_type}_{str(substance_type.value)}.pdf"

    data_frame_to_pdf(data_frame, os.path.join(output_directory, file_name))


def main():

    output_directory = "filtered_data"

    # Define the types of property which are of interest.
    properties_of_interest = [
        (Density, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (EnthalpyOfMixing, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
        (Density, SubstanceType.Binary),
    ]

    # Define the types of mixture which are of interest
    environment_types = ["alcohol_alcohol", "alcohol_ester", "ester_ester"]

    for environment_type in environment_types:

        environment_directory = os.path.join(output_directory, environment_type)
        os.makedirs(environment_directory, exist_ok=True)

        data_directory = os.path.join(
            "..",
            "data_availability",
            "data_by_environments",
            environment_type,
            "all_data",
        )

        for property_type in properties_of_interest:
            filter_data(data_directory, *property_type, environment_directory)


if __name__ == "__main__":
    main()
