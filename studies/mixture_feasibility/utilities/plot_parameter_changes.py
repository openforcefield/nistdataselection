import argparse
import os
from collections import defaultdict

import numpy
from matplotlib import pyplot
from openforcefield.typing.engines.smirnoff import ForceField


def plot_paramter_changes(original_parameters, optimized_parameters, output_path):

    parameter_attributes = ["epsilon", "rmin_half"]

    figure, axes = pyplot.subplots(2, 1, figsize=(8.0, 10))

    bar_height = 1.0 / len(original_parameters)
    study_colors = [
        "#BBDEFB",
        "#90CAF9",
        "#64B5F6",
        "#42A5F5",
        "#2196F3",
        "#1E88E5",
        "#1976D2",
        "#1565C0"
    ]

    study_names = [*next(iter(original_parameters.values())).keys()]

    for attribute_type, axis in zip(parameter_attributes, axes):

        x_values = numpy.arange(len(original_parameters))

        default_unit = getattr(
            next(iter(next(iter(original_parameters.values())).values())), attribute_type
        ).unit

        for smirks_index, smirks in enumerate(original_parameters):

            for study_index, study_name in enumerate(original_parameters[smirks]):

                original_parameter = getattr(
                    original_parameters[smirks][study_name], attribute_type
                )
                optimized_parameter = getattr(
                    optimized_parameters[smirks][study_name], attribute_type
                )

                initial_value = original_parameter.value_in_unit(default_unit)

                delta = optimized_parameter - original_parameter
                delta = delta.value_in_unit(default_unit)

                x_value = x_values[smirks_index] + study_index * bar_height

                axis.bar(
                    x_value,
                    initial_value + delta,
                    width=bar_height,
                    color=study_colors[study_index],
                    align="center",
                    label=study_name
                )
                axis.plot(
                    [x_values[smirks_index] - bar_height * 0.5, x_values[smirks_index] + 1 - bar_height * 1.5],
                    [initial_value, initial_value],
                    ":",
                    color="gray"
                )

        axis.set_ylabel(f"{attribute_type} ({default_unit})")

        axis.set_xticks(x_values + 0.5 - bar_height)
        axis.set_xticklabels([*original_parameters.keys()])

    handles, labels = figure.gca().get_legend_handles_labels()

    figure.tight_layout()

    figure.legend(
        handles,
        study_names,
        bbox_to_anchor=(0.1, 0.025, 0.8, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0.0,
        ncol=3,
    )
    figure.subplots_adjust(bottom=0.15)
    figure.savefig(output_path)


def main(fit_directories, force_field_name, output_path):

    # Find the values of the original and optimized parameters.
    original_parameters = defaultdict(dict)
    optimized_parameters = defaultdict(dict)

    for directory_path in fit_directories:

        original_force_field = ForceField(
            os.path.join(directory_path, "forcefield", force_field_name),
            allow_cosmetic_attributes=True,
        )

        optimized_force_field = ForceField(
            os.path.join(directory_path, "result", "optimize", force_field_name),
            allow_cosmetic_attributes=True,
        )

        directory_name = os.path.basename(os.path.dirname(directory_path))

        original_handler = original_force_field.get_parameter_handler("vdW")
        optimized_handler = optimized_force_field.get_parameter_handler("vdW")

        for parameter in original_handler.parameters:

            if not parameter.attribute_is_cosmetic("parameterize"):
                continue

            original_parameters[parameter.smirks][directory_name] = parameter
            optimized_parameters[parameter.smirks][
                directory_name
            ] = optimized_handler.parameters[parameter.smirks]

    # Plot the parameter changes
    plot_paramter_changes(original_parameters, optimized_parameters, output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot the changes in any optimized parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--directories",
        "-dirs",
        type=str,
        help="The paths to the fitting directories to include in the plot.",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--forcefield",
        "-f",
        type=str,
        help="The name of the optimized force field.",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="The path of the file to save the output to.",
        default="parameter_plots.png",
        required=False,
    )

    args = parser.parse_args()
    main(args.directories, args.forcefield, args.output)
