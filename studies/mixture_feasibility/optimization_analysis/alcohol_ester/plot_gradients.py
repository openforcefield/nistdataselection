import os

from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection.analysis.plotting import plot_gradient_per_environment
from nistdataselection.utils.utils import SubstanceType


def main():

    output_directory = "plots"
    os.makedirs(output_directory, exist_ok=True)

    # Define the names of the performed studies.
    study_names = [
        "h_mix_rho_x",
        "h_mix_rho_x_rho_pure",
        "h_mix_rho_x_rho_pure_h_vap",
        "h_mix_v_excess",
        "rho_pure_h_vap",
    ]

    # Define the types of property to plot.
    property_types = [
        (Density, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (Density, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
    ]

    plot_gradient_per_environment(property_types, study_names, output_directory, 0)
    plot_gradient_per_environment(property_types, study_names, output_directory, 1)


if __name__ == "__main__":
    main()
