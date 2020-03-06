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
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.properties import Density, EnthalpyOfMixing, ExcessMolarVolume

from nistdataselection.curation.filtering import (
    filter_by_smirks,
    filter_by_substance_composition,
    filter_by_temperature,
)
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import data_frame_to_pdf, property_to_snake_case


def main():

    output_directory = "filtered_data"
    os.makedirs(output_directory, exist_ok=True)

    # Determine which compounds were used during training.
    h_mix_v_excess_set = PhysicalPropertyDataSet.from_json(
        os.path.join(
            "..",
            "mixture_optimisation",
            "force_balance",
            "h_mix_binary_density",
            "targets",
            "mixture_data",
            "training_set.json",
        )
    )
    h_mix_rho_x_set = PhysicalPropertyDataSet.from_json(
        os.path.join(
            "..",
            "mixture_optimisation",
            "force_balance",
            "h_mix_v_excess",
            "targets",
            "mixture_data",
            "training_set.json",
        )
    )

    training_smiles = [
        *((x.smiles for x in y) for y in h_mix_v_excess_set.substances),
        *((x.smiles for x in y) for y in h_mix_rho_x_set.substances),
    ]
    training_smiles = set(tuple(sorted(x)) for x in training_smiles)

    for pair_types in ["alcohol_alcohol", "alcohol_ester", "ester_ester"]:

        pair_directory = os.path.join(output_directory, pair_types)
        os.makedirs(pair_directory, exist_ok=True)

        for property_type, substance_type in [
            (EnthalpyOfMixing, SubstanceType.Binary),
            (ExcessMolarVolume, SubstanceType.Binary),
            (Density, SubstanceType.Binary),
        ]:

            # Load in the data set
            folder_name = f"all_{pair_types}_data"

            data_frame = load_processed_data_set(
                os.path.join("..", "data_availability", folder_name),
                property_type,
                substance_type,
            )

            # Filter to be close to ambient.
            data_frame = filter_by_temperature(
                data_frame, 290.0 * unit.kelvin, 305 * unit.kelvin
            )
            # Filter out the training set
            data_frame = filter_by_substance_composition(
                data_frame, None, training_smiles
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

            # Save the filtered set.
            save_processed_data_set(
                pair_directory, data_frame, property_type, substance_type,
            )

            property_type = property_to_snake_case(property_type)
            file_name = f"{property_type}_{str(substance_type.value)}.pdf"

            data_frame_to_pdf(data_frame, os.path.join(pair_directory, file_name))


if __name__ == "__main__":
    main()
