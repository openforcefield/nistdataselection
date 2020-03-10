import os

import numpy
from evaluator import unit
from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)
from matplotlib import pyplot

from nistdataselection.processing import load_processed_data_set
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import property_to_title
from studies.mixture_feasibility.benchmarking.analysis.statistics import (
    Statistics,
    compute_bootstrapped_statistics,
    compute_statistic_unit,
)


def plot_estimated_vs_experiment(
    root_results_directory,
    study_names,
    property_types,
    output_directory,
    dots_per_inch=200,
    figure_size=5.0,
):

    figure, axes = pyplot.subplots(
        nrows=len(property_types),
        ncols=len(study_names),
        sharey="row",
        dpi=dots_per_inch,
        figsize=(figure_size * len(study_names), figure_size * len(property_types)),
    )

    for row_index, (property_type, substance_type) in enumerate(property_types):

        default_unit = property_type.default_unit()

        title = property_to_title(property_type, substance_type, default_unit)
        title = f"Experimental {title}"

        axes[row_index][0].set_ylabel(title)

        for column_index, study_name in enumerate(study_names):

            data_frame = load_processed_data_set(
                os.path.join(root_results_directory, study_name),
                property_type,
                substance_type,
            )

            measured_values = data_frame[
                f"Target {property_type.__name__} Value ({default_unit:~})"
            ]
            estimated_values = data_frame[
                f"Estimated {property_type.__name__} Value ({default_unit:~})"
            ]

            min_value = numpy.min([*measured_values, *estimated_values]) * 0.95
            max_value = numpy.max([*measured_values, *estimated_values]) * 1.05

            estimated_uncertainties = data_frame[
                f"Estimated {property_type.__name__} Uncertainty ({default_unit:~})"
            ]

            means, _, ci = compute_bootstrapped_statistics(
                measured_values,
                estimated_values,
                statistics=[Statistics.RMSE, Statistics.R2],
            )

            axis = axes[row_index][column_index]

            axis.text(
                0.03,
                0.90,
                f"$R^2 = {means[Statistics.R2]:.2f}_"
                f"{{{ci[Statistics.R2][0]:.2f}}}^"
                f"{{{ci[Statistics.R2][1]:.2f}}}$",
                transform=axis.transAxes,
            )
            axis.text(
                0.03,
                0.75,
                f"$RMSE = {means[Statistics.RMSE]:.2f}_"
                f"{{{ci[Statistics.RMSE][0]:.2f}}}^"
                f"{{{ci[Statistics.RMSE][1]:.2f}}}$",
                transform=axis.transAxes,
            )

            axis.errorbar(
                x=estimated_values,
                y=measured_values,
                xerr=estimated_uncertainties,
                fmt="x",
                label=study_name,
                markersize=5.0,
            )

            axis.set_xlabel(study_name)

            axis.set_xlim(min_value, max_value)
            axis.set_ylim(min_value, max_value)

    figure.savefig(
        os.path.join(output_directory, f"estimated_vs_experiment.pdf"),
        bbox_inches="tight",
    )
    figure.savefig(
        os.path.join(output_directory, f"estimated_vs_experiment.png"),
        bbox_inches="tight",
    )


def plot_per_property_statistic(
    root_results_directory,
    study_names,
    property_types,
    statistics_type,
    output_directory,
    dots_per_inch=200,
    figure_size=5.0,
):

    figure, axes = pyplot.subplots(
        nrows=1,
        ncols=len(property_types),
        dpi=dots_per_inch,
        figsize=(figure_size * len(property_types), figure_size),
    )

    for column_index, (property_type, substance_type) in enumerate(property_types):

        default_unit = property_type.default_unit()

        for study_index, study_name in enumerate(study_names):

            data_frame = load_processed_data_set(
                os.path.join(root_results_directory, study_name),
                property_type,
                substance_type,
            )

            measured_values = data_frame[
                f"Target {property_type.__name__} Value ({default_unit:~})"
            ]
            estimated_values = data_frame[
                f"Estimated {property_type.__name__} Value ({default_unit:~})"
            ]

            means, _, ci = compute_bootstrapped_statistics(
                measured_values, estimated_values, statistics=[statistics_type]
            )

            bar_values = means[statistics_type]
            bar_errors = [
                means[statistics_type] - ci[statistics_type][0],
                ci[statistics_type][1] - means[statistics_type],
            ]

            axes[column_index].bar(
                x=study_index,
                height=bar_values,
                yerr=numpy.array([bar_errors]).T,
                align="center",
                label=study_name,
            )

        axes[column_index].set_title(property_to_title(property_type, substance_type))

        statistic_unit = compute_statistic_unit(default_unit, statistics_type)

        unit_string = (
            f" {statistic_unit:~}"
            if statistic_unit is not None and statistic_unit != unit.dimensionless
            else ""
        )

        axes[column_index].set_ylabel(f"{str(statistics_type.value)}{unit_string}")

    handles, labels = axes[-1].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        bbox_to_anchor=(0.25, 0.025, 0.5, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0.0,
        ncol=min(5, len(study_names)),
    )
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.2)

    figure.savefig(os.path.join(output_directory, f"{str(statistics_type.value)}.pdf"))


def plot_per_statistics_per_mixture_type_bar(
    root_results_directory,
    study_names,
    property_types,
    statistics_type,
    mixture_types,
    training_restrictions,
    output_directory,
    dots_per_inch=200,
    figure_size=5.0,
):

    for training_restriction in training_restrictions:

        figure, outer_axes = pyplot.subplots(
            nrows=len(mixture_types),
            ncols=1,
            dpi=dots_per_inch,
            figsize=(figure_size * len(property_types), figure_size * len(mixture_types)),
        )

        axis = None

        for row_index, mixture_type in enumerate(mixture_types):

            outer_axis = outer_axes[row_index]

            outer_axis.set_title(mixture_type.title())
            outer_axis.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
            outer_axis._frameon = False

        for row_index, mixture_type in enumerate(mixture_types):

            for column_index, (property_type, substance_type) in enumerate(property_types):

                axis_index = column_index + row_index * len(property_types) + 1
                axis = figure.add_subplot(len(mixture_types), len(property_types), axis_index)

                default_unit = property_type.default_unit()

                for study_index, study_name in enumerate(study_names):

                    data_frame = load_processed_data_set(
                        os.path.join(root_results_directory, study_name, mixture_type, training_restriction),
                        property_type,
                        substance_type,
                    )

                    if len(data_frame) == 0:
                        continue

                    measured_values = data_frame[
                        f"Target {property_type.__name__} Value ({default_unit:~})"
                    ]
                    estimated_values = data_frame[
                        f"Estimated {property_type.__name__} Value ({default_unit:~})"
                    ]

                    means, _, ci = compute_bootstrapped_statistics(
                        measured_values, estimated_values, statistics=[statistics_type]
                    )

                    bar_values = means[statistics_type]
                    bar_errors = [
                        means[statistics_type] - ci[statistics_type][0],
                        ci[statistics_type][1] - means[statistics_type],
                    ]

                    axis.bar(
                        x=study_index,
                        height=bar_values,
                        yerr=numpy.array([bar_errors]).T,
                        align="center",
                        label=study_name,
                    )

                    axis.set_xlabel(
                        property_to_title(property_type, substance_type)
                    )

                statistic_unit = compute_statistic_unit(default_unit, statistics_type)

                unit_string = (
                    f" {statistic_unit:~}"
                    if statistic_unit is not None and statistic_unit != unit.dimensionless
                    else ""
                )

                axis.set_ylabel(f"{str(statistics_type.value)}{unit_string}")

        if axis is not None:

            handles, labels = axis.get_legend_handles_labels()

            figure.legend(
                handles,
                labels,
                bbox_to_anchor=(0.2, 0.025, 0.6, 0.2),
                loc="lower left",
                mode="expand",
                borderaxespad=0.0,
                ncol=min(5, len(study_names)),
            )

        figure.tight_layout()
        figure.subplots_adjust(bottom=0.1, hspace=0.25)

        figure.savefig(os.path.join(output_directory, f"{training_restriction}_{str(statistics_type.value)}.pdf"))
        pyplot.close(figure)


def plot_per_statistics_per_mixture_type(
    root_results_directory,
    study_names,
    property_types,
    statistics_type,
    mixture_types,
    training_restrictions,
    output_directory,
    dots_per_inch=200,
    figure_size=5.0,
):

    for training_restriction in training_restrictions:

        figure, outer_axes = pyplot.subplots(
            nrows=len(mixture_types),
            ncols=1,
            dpi=dots_per_inch,
            figsize=(figure_size * len(property_types), figure_size * len(mixture_types)),
        )

        axis = None

        for row_index, mixture_type in enumerate(mixture_types):

            outer_axis = outer_axes[row_index]

            outer_axis.set_title(mixture_type.title())
            outer_axis.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
            outer_axis._frameon = False

        for row_index, mixture_type in enumerate(mixture_types):

            for column_index, (property_type, substance_type) in enumerate(property_types):

                axis_index = column_index + row_index * len(property_types) + 1
                axis = figure.add_subplot(len(mixture_types), len(property_types), axis_index)

                default_unit = property_type.default_unit()

                for study_index, study_name in enumerate(study_names):

                    data_frame = load_processed_data_set(
                        os.path.join(root_results_directory, study_name, mixture_type, training_restriction),
                        property_type,
                        substance_type,
                    )

                    if len(data_frame) == 0:
                        continue

                    measured_values = data_frame[
                        f"Target {property_type.__name__} Value ({default_unit:~})"
                    ]
                    estimated_values = data_frame[
                        f"Estimated {property_type.__name__} Value ({default_unit:~})"
                    ]

                    min_value = numpy.min([*measured_values, *estimated_values]) * 0.95
                    max_value = numpy.max([*measured_values, *estimated_values]) * 1.05

                    estimated_uncertainties = data_frame[
                        f"Estimated {property_type.__name__} Uncertainty ({default_unit:~})"
                    ]

                    axis.errorbar(
                        x=estimated_values,
                        y=measured_values,
                        xerr=estimated_uncertainties,
                        fmt="x",
                        label=study_name,
                        markersize=5.0,
                    )

                    axis.set_xlabel(
                        property_to_title(property_type, substance_type)
                    )

                    axis.set_xlim(min_value, max_value)
                    axis.set_ylim(min_value, max_value)

        if axis is not None:

            handles, labels = axis.get_legend_handles_labels()

            figure.legend(
                handles,
                labels,
                bbox_to_anchor=(0.2, 0.025, 0.6, 0.2),
                loc="lower left",
                mode="expand",
                borderaxespad=0.0,
                ncol=min(5, len(study_names)),
            )

        figure.tight_layout()
        figure.subplots_adjust(bottom=0.1, hspace=0.25)

        figure.savefig(os.path.join(output_directory, f"{training_restriction}_{str(statistics_type.value)}_expt.pdf"))
        pyplot.close(figure)


def main():

    output_directory = "plots"
    os.makedirs(output_directory, exist_ok=True)

    study_names = [
        "openff-1.0.0",
        "h_mix_rho_x",
        "h_mix_rho_x_rho_pure",
        "h_mix_v_excess",
        "rho_pure_h_vap",
    ]

    property_types = [
        (Density, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (Density, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
    ]

    mixture_types = ["alcohol_alcohol", "alcohol_ester", "ester_ester"]
    training_restrictions = ["both_in_training", "one_in_training", "not_in_training"]

    # root_results_directory = os.path.join("..", "results", "all_results")
    # plot_estimated_vs_experiment(
    #     root_results_directory, study_names, property_types, output_directory
    # )
    # plot_per_property_statistic(
    #     root_results_directory,
    #     study_names,
    #     property_types,
    #     Statistics.RMSE,
    #     output_directory,
    # )

    root_results_directory = os.path.join("..", "results", "partitioned_results")

    property_types = [
        (Density, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
    ]

    plot_per_statistics_per_mixture_type(
        root_results_directory,
        study_names,
        property_types,
        Statistics.RMSE,
        mixture_types,
        training_restrictions,
        output_directory
    )
    plot_per_statistics_per_mixture_type_bar(
        root_results_directory,
        study_names,
        property_types,
        Statistics.RMSE,
        mixture_types,
        training_restrictions,
        output_directory
    )


if __name__ == "__main__":
    main()
