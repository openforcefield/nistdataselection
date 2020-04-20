import os
from glob import glob

from evaluator.properties import Density, EnthalpyOfMixing, EnthalpyOfVaporization

from nistdataselection.analysis.processing import extract_gradients
from nistdataselection.utils import SubstanceType


def main():

    output_directory = "gradients"
    os.makedirs(output_directory, exist_ok=True)

    # Define the names of the performed studies.
    study_names = [
        "h_mix_rho_x",
        "h_mix_rho_x_rho_pure",
        "h_mix_rho_x_rho_pure_h_vap",
        "rho_pure_h_vap",
    ]

    # Define the names of the properties which were benchmarked.
    property_types = [
        (Density, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (Density, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
    ]

    # Extract gradients for each of the different environments.
    partitioned_directory = os.path.join("partitioned_data")

    all_environments = set(
        tuple(os.path.basename(x).split("_"))
        for y in study_names
        for x in glob(os.path.join(partitioned_directory, y, "*"))
    )

    environments = {
        SubstanceType.Pure: [x for x in all_environments if len(x) == 1],
        SubstanceType.Binary: [x for x in all_environments if len(x) == 2],
    }

    gradients_per_environment = extract_gradients(
        partitioned_directory,
        property_types,
        study_names,
        environments,
        partition_by_composition=True,
    )
    gradients_per_environment.to_csv(
        os.path.join(output_directory, "per_composition.csv"), index=False
    )


if __name__ == "__main__":
    main()
