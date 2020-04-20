import os

from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection.analysis.plotting import (
    plot_estimated_vs_reference,
    plot_statistic,
    plot_statistic_per_environment,
)
from nistdataselection.analysis.statistics import Statistics
from nistdataselection.utils import SubstanceType


def main():

    output_directory = "plots"
    os.makedirs(output_directory, exist_ok=True)

    # Define the types of property to plot.
    property_types = [
        (Density, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (Density, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
    ]

    study_names = [
        "openff-1.0.0",
        "h_mix_rho_x",
        "h_mix_rho_x_rho_pure",
        "h_mix_rho_x_rho_pure_h_vap",
        "rho_pure_h_vap",
    ]

    plot_estimated_vs_reference(property_types, study_names, output_directory)

    # Define the statistics to plot.
    statistics = [Statistics.RMSE, Statistics.R2]

    # Plot a summary of each statistic.
    plot_statistic(statistics, output_directory)

    statistics = [Statistics.RMSE]

    # # Plot the statistics per environment
    pure_properties = [x for x in property_types if x[1] == SubstanceType.Pure]
    plot_statistic_per_environment(pure_properties, statistics, output_directory)

    binary_properties = [x for x in property_types if x[1] == SubstanceType.Binary]
    plot_statistic_per_environment(
        binary_properties, statistics, output_directory, per_composition=True
    )


if __name__ == "__main__":
    main()
