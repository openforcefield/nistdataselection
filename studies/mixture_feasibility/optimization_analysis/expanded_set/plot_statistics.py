import os

from evaluator.properties import Density, EnthalpyOfMixing, EnthalpyOfVaporization

from nistdataselection.analysis.plotting import (
    plot_objective_per_iteration,
    plot_statistic_per_iteration,
)
from nistdataselection.analysis.statistics import Statistics
from nistdataselection.utils import SubstanceType


def main():

    output_directory = "plots"
    os.makedirs(output_directory, exist_ok=True)

    # Define the names of the performed studies.
    study_names = [
        "h_mix_rho_x",
        "h_mix_rho_x_rho_pure",
        "h_mix_rho_x_rho_pure_h_vap",
        "rho_pure_h_vap",
    ]

    # Define the types of property to plot.
    property_types = [
        (Density, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (Density, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
    ]

    # Plot a summary of each statistic per iteration.
    for property_type in property_types:
        plot_statistic_per_iteration(
            property_type,
            Statistics.RMSE,
            output_directory,
            per_composition=property_type[1] == SubstanceType.Binary,
            max_iterations=5,
        )

    # Plot the statistics per environment
    plot_objective_per_iteration(study_names, output_directory)


if __name__ == "__main__":
    main()
