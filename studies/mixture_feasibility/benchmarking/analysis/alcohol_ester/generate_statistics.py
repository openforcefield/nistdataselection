import os
from glob import glob

from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection.analysis.processing import generate_statistics
from nistdataselection.utils import SubstanceType


def main():

    output_directory = "statistics"
    os.makedirs(output_directory, exist_ok=True)

    # Define how many processes to parallelize the computations over.
    n_processes = 4
    # Define how many bootstrap iterations to perform when computing
    # the statistics.
    n_bootstrap_iterations = 2000

    # Define the names of the performed studies.
    study_names = [
        "h_mix_rho_x",
        "h_mix_rho_x_rho_pure",
        "h_mix_rho_x_rho_pure_h_vap",
        "h_mix_v_excess",
        "openff-1.0.0",
        "rho_pure_h_vap",
    ]

    # Define the names of the properties which were benchmarked.
    property_types = [
        (Density, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (Density, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
    ]

    # Generate statistics for the full benchmark set.
    all_results_directory = os.path.join("all_data")

    all_statistics = generate_statistics(
        all_results_directory,
        property_types,
        study_names,
        None,
        partition_by_composition=False,
        bootstrap_iterations=n_bootstrap_iterations,
        n_processes=n_processes,
    )
    all_statistics.to_csv(
        os.path.join(output_directory, "all_statistics.csv"), index=False
    )

    # Generate statistics for each of the different environments.
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

    statistics_per_environment = generate_statistics(
        partitioned_directory,
        property_types,
        study_names,
        environments,
        partition_by_composition=False,
        bootstrap_iterations=n_bootstrap_iterations,
        n_processes=n_processes,
    )
    statistics_per_environment.to_csv(
        os.path.join(output_directory, "per_environment.csv"), index=False
    )

    # Partition the mixture properties by their compositions as well
    # as their environments
    mixture_types = [x for x in property_types if x[1] == SubstanceType.Binary]

    statistics_per_environment = generate_statistics(
        partitioned_directory,
        mixture_types,
        study_names,
        environments,
        partition_by_composition=True,
        bootstrap_iterations=n_bootstrap_iterations,
        n_processes=n_processes,
    )
    statistics_per_environment.to_csv(
        os.path.join(output_directory, "per_composition.csv"), index=False
    )


if __name__ == "__main__":
    main()
