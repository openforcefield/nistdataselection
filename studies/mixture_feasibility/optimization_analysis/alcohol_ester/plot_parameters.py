import os

from nistdataselection.analysis.plotting import plot_parameter_changes


def main():

    output_directory = "plots"
    os.makedirs(output_directory, exist_ok=True)

    # Define the names of the performed studies.
    study_names = [
        "h_mix_rho_x",
        "h_mix_rho_x_rho_pure",
        # "h_mix_rho_x_rho_pure_h_vap",
        "rho_pure_h_vap",
    ]

    parameter_smirks = [
        "[#1:1]-[#6X4]",
        "[#6:1]",
        "[#6X4:1]",
        "[#8:1]",
        "[#8X2H0+0:1]",
        "[#8X2H1+0:1]",
    ]

    plot_parameter_changes(
        "openff-1.0.0.offxml",
        "refit_force_fields",
        study_names,
        parameter_smirks,
        output_directory,
    )


if __name__ == "__main__":
    main()
