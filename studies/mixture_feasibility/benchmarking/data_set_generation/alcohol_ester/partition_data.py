"""A script to partition the data into

- a `not_in_training` folder, where none of the components the data was
  measured for appeared in the training set.
- a `one_in_training` folder, where one of the components the data was
  measured for appeared in the training set (mixture data only).
- a `both_in_training` folder where both of the components the data was
  measured for appeared in the training set (mixture data only).

All outputs will be stored in the `partitioned_data` folder.
"""
import os

from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection.processing import load_processed_data_set
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.pandas import data_frame_to_smiles_tuples
from nistdataselection.utils.utils import data_frame_to_pdf, property_to_snake_case


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
            "..",
            "pure_mixture_optimisation",
            "force_balance",
            "alcohol_ester",
            "h_mix_rho_x_rho_pure_h_vap",
            "targets",
            "mixture_data",
            "training_set.json",
        )
    ).to_pandas()

    training_smiles = data_frame_to_smiles_tuples(training_set)
    training_smiles = set(x for y in training_smiles for x in y)

    return training_smiles


def main():

    root_output_directory = "partitioned_data"

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

    # Find all of the substances which appeared in the training set
    training_smiles = find_training_smiles()

    for environment_type in environment_types:

        output_directory = os.path.join(root_output_directory, environment_type)
        os.makedirs(output_directory, exist_ok=True)

        for property_type, substance_type in properties_of_interest:

            full_data_frame = load_processed_data_set(
                os.path.join("filtered_data", environment_type),
                property_type,
                substance_type,
            )

            property_type = property_to_snake_case(property_type)
            file_name = f"{property_type}_{str(substance_type.value)}"

            # Extract properties where neither component appears in
            # in the training set.
            if substance_type == SubstanceType.Binary:

                data_frame = full_data_frame[
                    ~full_data_frame["Component 1"].isin(training_smiles)
                    & ~full_data_frame["Component 2"].isin(training_smiles)
                ]

            elif substance_type == SubstanceType.Pure:

                data_frame = full_data_frame[
                    ~full_data_frame["Component 1"].isin(training_smiles)
                ]

            else:

                raise NotImplementedError()

            base_directory = os.path.join(output_directory, "not_in_training")
            os.makedirs(base_directory, exist_ok=True)

            data_frame.to_csv(
                os.path.join(base_directory, file_name + ".csv"), index=False
            )
            data_frame_to_pdf(
                data_frame, os.path.join(base_directory, file_name + ".pdf")
            )

            if substance_type == SubstanceType.Pure:
                continue

            # Extract properties where both components appear in
            # in the training set.
            data_frame = full_data_frame[
                full_data_frame["Component 1"].isin(training_smiles)
                & full_data_frame["Component 2"].isin(training_smiles)
            ]

            base_directory = os.path.join(output_directory, "both_in_training")
            os.makedirs(base_directory, exist_ok=True)

            data_frame.to_csv(
                os.path.join(base_directory, file_name + ".csv"), index=False
            )
            data_frame_to_pdf(
                data_frame, os.path.join(base_directory, file_name + ".pdf")
            )

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

            data_frame.to_csv(
                os.path.join(base_directory, file_name + ".csv"), index=False
            )
            data_frame_to_pdf(
                data_frame, os.path.join(base_directory, file_name + ".pdf")
            )


if __name__ == "__main__":
    main()
