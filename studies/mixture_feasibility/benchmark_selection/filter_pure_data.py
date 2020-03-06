"""A script to filter out all of the training set
compounds from the available pure data.
"""
import os

import pandas
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.properties import Density, EnthalpyOfVaporization

from nistdataselection.curation.filtering import filter_by_smiles
from nistdataselection.processing import save_processed_data_set
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import data_frame_to_pdf


def main():

    output_directory = "filtered_data"
    os.makedirs(output_directory, exist_ok=True)

    # Determine which compounds were used during training.
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

    training_smiles = set(x.smiles for y in pure_training_set.substances for x in y)

    # Load in the Hvap data
    h_vap_data_frame = pandas.read_csv(
        os.path.join(
            "..", "data_availability", "sourced_h_vap_data", "alcohol_ester_h_vap.csv"
        )
    )
    h_vap_data_frame = filter_by_smiles(
        h_vap_data_frame, smiles_to_include=None, smiles_to_exclude=training_smiles,
    )
    save_processed_data_set(
        output_directory, h_vap_data_frame, EnthalpyOfVaporization, SubstanceType.Pure,
    )
    data_frame_to_pdf(
        h_vap_data_frame,
        os.path.join(output_directory, "enthalpy_of_vaporization_pure.pdf"),
    )

    # Pull out the smiles patterns for which there exists Hvap data.
    test_smiles = [*h_vap_data_frame["Component 1"]]

    # Load in the density data
    density_data_frame = pandas.read_csv(
        os.path.join(
            "..", "data_availability", "all_alcohol_ester_data", "density_pure.csv"
        )
    )
    density_data_frame = filter_by_smiles(
        density_data_frame, smiles_to_include=test_smiles, smiles_to_exclude=None
    )
    save_processed_data_set(
        output_directory, density_data_frame, Density, SubstanceType.Pure,
    )
    data_frame_to_pdf(
        h_vap_data_frame, os.path.join(output_directory, "density_pure.pdf")
    )


if __name__ == "__main__":
    main()
