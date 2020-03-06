"""This script partitions all available ester(/acid)-ester(/acid)
mixture data into mixture where both components where in the training
set, one component was in the training set and neither component was
in the training set.
"""
import os

from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.properties import Density, EnthalpyOfMixing, ExcessMolarVolume

from nistdataselection.curation.filtering import filter_by_checkmol
from nistdataselection.processing import load_processed_data_set
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import data_frame_to_pdf, property_to_snake_case

chemical_environment_codes = {
    "caboxylic_acid": "076",
    "ester": "078",
}


def main():
    output_directory = "partitioned_ester_ester_data"
    os.makedirs(output_directory, exist_ok=True)

    # Find those alcohols which were included in the training set
    pure_training_set = PhysicalPropertyDataSet.from_json(
        os.path.join(
            "..",
            "pure_optimisation",
            "force_balance",
            "targets",
            "pure_data",
            "training_set.json",
        )
    )
    pure_alcohol_set = filter_by_checkmol(
        pure_training_set.to_pandas(),
        [
            chemical_environment_codes["caboxylic_acid"],
            chemical_environment_codes["ester"],
        ],
    )

    training_smiles = {*pure_alcohol_set["Component 1"]}

    for property_type, substance_type in [
        (Density, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
    ]:

        full_data_frame = load_processed_data_set(
            os.path.join("filtered_data", "ester_ester"), property_type, substance_type,
        )

        property_type = property_to_snake_case(property_type)

        file_name = f"{property_type}_{str(substance_type.value)}"

        # Extract properties where both components appear in
        # in the training set.
        data_frame = full_data_frame[
            full_data_frame["Component 1"].isin(training_smiles)
            & full_data_frame["Component 2"].isin(training_smiles)
        ]

        base_directory = os.path.join(output_directory, "both_in_training")
        os.makedirs(base_directory, exist_ok=True)

        data_frame.to_csv(os.path.join(base_directory, file_name + ".csv"), index=False)
        data_frame_to_pdf(data_frame, os.path.join(base_directory, file_name + ".pdf"))

        # Extract properties where only one component appears in
        # in the training set.
        data_frame = full_data_frame[
            (
                full_data_frame["Component 1"].isin(training_smiles)
                & ~full_data_frame["Component 2"].isin(training_smiles)
            )
            | (
                ~full_data_frame["Component 1"].isin(training_smiles)
                & full_data_frame["Component 2"].isin(training_smiles)
            )
        ]

        base_directory = os.path.join(output_directory, "one_in_training")
        os.makedirs(base_directory, exist_ok=True)

        data_frame.to_csv(os.path.join(base_directory, file_name + ".csv"), index=False)
        data_frame_to_pdf(data_frame, os.path.join(base_directory, file_name + ".pdf"))

        # Extract properties where neither component appears in
        # in the training set.
        data_frame = full_data_frame[
            ~full_data_frame["Component 1"].isin(training_smiles)
            & ~full_data_frame["Component 2"].isin(training_smiles)
        ]

        base_directory = os.path.join(output_directory, "neither_in_training")
        os.makedirs(base_directory, exist_ok=True)

        data_frame.to_csv(os.path.join(base_directory, file_name + ".csv"), index=False)
        data_frame_to_pdf(data_frame, os.path.join(base_directory, file_name + ".pdf"))


if __name__ == "__main__":
    main()
