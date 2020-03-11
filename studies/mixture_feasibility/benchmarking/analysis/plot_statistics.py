import os
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum

import numpy
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
)


class PlotType(Enum):

    Scatter = "Scatter"
    Bar = "Bar"


def determine_axis_limits(data, share_x=True, share_y=True, square_axis=True):

    # Validate the data.
    minimum_n_columns = min(len(x) for x in data.values())
    maximum_n_columns = max(len(x) for x in data.values())

    assert minimum_n_columns == maximum_n_columns

    n_rows = len(data)
    n_columns = minimum_n_columns

    # Determine the row and column axis limits
    minimum_x_limits = numpy.empty((n_rows, n_columns))
    maximum_x_limits = numpy.empty((n_rows, n_columns))
    minimum_y_limits = numpy.empty((n_rows, n_columns))
    maximum_y_limits = numpy.empty((n_rows, n_columns))

    minimum_x_limits[:] = numpy.nan
    maximum_x_limits[:] = numpy.nan
    minimum_y_limits[:] = numpy.nan
    maximum_y_limits[:] = numpy.nan

    for row_index, columns in enumerate(data.values()):

        for column_index, all_series in enumerate(columns.values()):

            min_x_values = []
            max_x_values = []

            min_y_values = []
            max_y_values = []

            for series in all_series.values():

                x_values, x_std, y_values, y_std = series

                if not isinstance(x_values, Iterable):
                    x_values = numpy.array([x_values])
                if not isinstance(y_values, Iterable):
                    y_values = numpy.array([y_values])

                if not isinstance(x_std, Iterable):

                    if x_std is None:
                        x_std = numpy.zeros(x_values.shape)
                    else:
                        x_std = numpy.array([x_std])

                if not isinstance(y_std, Iterable):

                    if y_std is None:
                        y_std = numpy.zeros(y_values.shape)
                    else:
                        y_std = numpy.array([y_std])

                min_x_values.extend(x_values - x_std)
                max_x_values.extend(x_values + x_std)

                min_y_values.extend(y_values - y_std)
                max_y_values.extend(y_values + y_std)

            if (
                len(min_x_values) == 0
                or len(max_x_values) == 0
                or len(min_y_values) == 0
                or len(max_y_values) == 0
            ):
                continue

            minimum_x_limits[row_index, column_index] = (
                numpy.nanmin(min_x_values) * 0.95
            )
            maximum_x_limits[row_index, column_index] = (
                numpy.nanmax(max_x_values) * 1.05
            )

            minimum_y_limits[row_index, column_index] = (
                numpy.nanmin(min_y_values) * 0.95
            )
            maximum_y_limits[row_index, column_index] = (
                numpy.nanmax(max_y_values) * 1.05
            )

    if share_x:

        for x in range(n_columns):

            minimum_x_limits[:, x] = numpy.nanmin(minimum_x_limits[:, x])
            maximum_x_limits[:, x] = numpy.nanmax(maximum_x_limits[:, x])

    if share_y:

        for y in range(n_rows):

            minimum_y_limits[y, :] = numpy.nanmin(minimum_y_limits[y, :])
            maximum_y_limits[y, :] = numpy.nanmax(maximum_y_limits[y, :])

    if square_axis:

        for x in range(n_columns):

            for y in range(n_rows):

                minimum_limit = numpy.nanmin(
                    [minimum_x_limits[y, x], minimum_y_limits[y, x]]
                )
                maximum_limit = numpy.nanmax(
                    [maximum_x_limits[y, x], maximum_y_limits[y, x]]
                )

                minimum_x_limits[y, x] = minimum_limit
                minimum_y_limits[y, x] = minimum_limit

                maximum_x_limits[y, x] = maximum_limit
                maximum_y_limits[y, x] = maximum_limit

    return minimum_x_limits, maximum_x_limits, minimum_y_limits, maximum_y_limits


def plot_data(
    plot_type,
    data,
    statistics=None,
    x_axis_label=None,
    y_axis_label=None,
    include_row_title=True,
    include_column_title=True,
    include_legend=True,
    share_x=True,
    share_y=True,
    square_axis=True,
    dots_per_inch=200,
    sub_plot_size=5.0,
    marker_format="x",
    marker_size=5.0,
):

    # Validate and determine the shape of the data.
    minimum_n_columns = min(len(x) for x in data.values())
    maximum_n_columns = max(len(x) for x in data.values())

    assert minimum_n_columns == maximum_n_columns

    n_rows = len(data)
    n_columns = minimum_n_columns

    # Create the figure and axes
    figure, outer_axes = pyplot.subplots(
        nrows=n_rows,
        ncols=1,
        dpi=dots_per_inch,
        figsize=(sub_plot_size * n_columns, sub_plot_size * n_rows * 1.05),
    )

    if include_row_title:

        # Optionally add row titles.
        for row_index, row_label in enumerate(data):

            if n_rows > 1:
                outer_axis = outer_axes[row_index]
            else:
                outer_axis = outer_axes

            outer_axis.set_title(row_label, fontsize=14, pad=35)

            # Turn off axis lines and ticks of the outer subplot
            outer_axis.spines["top"].set_color("none")
            outer_axis.spines["bottom"].set_color("none")
            outer_axis.spines["left"].set_color("none")
            outer_axis.spines["right"].set_color("none")
            outer_axis.tick_params(
                labelcolor="w", top=False, bottom=False, left=False, right=False
            )

        figure.subplots_adjust(hspace=0.4)

    (
        minimum_x_limits,
        maximum_x_limits,
        minimum_y_limits,
        maximum_y_limits,
    ) = determine_axis_limits(data, share_x, share_y, square_axis)

    for row_index, columns in enumerate(data.values()):

        for column_index, (column_label, all_series) in enumerate(columns.items()):

            axis_index = column_index + row_index * n_columns
            axis = figure.add_subplot(n_rows, n_columns, axis_index + 1)

            if include_column_title:
                axis.set_title(column_label)

            if row_index == len(data) - 1 or include_row_title:
                axis.set_xlabel(x_axis_label)

            if column_index == 0:
                axis.set_ylabel(y_axis_label)

            if (
                not numpy.isnan(minimum_x_limits[row_index, column_index])
                and not numpy.isnan(maximum_x_limits[row_index, column_index])
                and not plot_type == PlotType.Bar
            ):
                axis.set_xlim(
                    minimum_x_limits[row_index, column_index],
                    maximum_x_limits[row_index, column_index],
                )
            if not numpy.isnan(
                minimum_y_limits[row_index, column_index]
            ) and not numpy.isnan(maximum_y_limits[row_index, column_index]):

                if plot_type == PlotType.Bar:
                    minimum_y_limits[row_index, column_index] = 0.0

                axis.set_ylim(
                    minimum_y_limits[row_index, column_index],
                    maximum_y_limits[row_index, column_index],
                )

            all_x_values = []
            all_y_values = []

            for series_label, series in all_series.items():

                x_values, x_std, y_values, y_std = series

                if not isinstance(x_values, Iterable) and x_values is not None:
                    x_values = numpy.array([x_values])
                if not isinstance(x_std, Iterable) and x_std is not None:
                    x_std = numpy.array([x_std])
                if not isinstance(y_values, Iterable) and y_values is not None:
                    y_values = numpy.array([y_values])
                if not isinstance(y_std, Iterable) and y_std is not None:
                    y_std = numpy.array([y_std])

                all_x_values.extend(x_values)
                all_y_values.extend(y_values)

                if plot_type == PlotType.Scatter:

                    axis.errorbar(
                        x=x_values,
                        y=y_values,
                        xerr=x_std,
                        yerr=y_std,
                        fmt=marker_format,
                        label=series_label,
                        markersize=marker_size,
                    )

                elif plot_type == PlotType.Bar:

                    bar_errors = [
                        y_values - y_std[0],
                        y_std[1] - y_values,
                    ]

                    axis.bar(
                        x=x_values,
                        height=y_values,
                        yerr=numpy.array(bar_errors),
                        align="center",
                        label=series_label,
                    )

            if statistics is None:
                continue

            all_x_values = numpy.array(all_x_values)
            all_y_values = numpy.array(all_y_values)

            if len(all_x_values) == 0 or len(all_y_values) == 0:
                continue

            means, _, ci = compute_bootstrapped_statistics(
                all_x_values,
                all_y_values,
                statistics=statistics,
                bootstrap_iterations=100,
            )

            for statistic_index, statistic in enumerate(statistics):

                axis.text(
                    0.03,
                    0.90 - statistic_index * 0.15,
                    f"${statistic.value} = {means[statistic]:.2f}_"
                    f"{{{ci[statistic][0]:.2f}}}^"
                    f"{{{ci[statistic][1]:.2f}}}$",
                    transform=axis.transAxes,
                )

    if include_legend:

        handles, labels = figure.gca().get_legend_handles_labels()

        figure.legend(
            handles,
            labels,
            bbox_to_anchor=(0.25, 0.025, 0.5, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0.0,
            ncol=5,
        )
        # figure.tight_layout()

        if plot_type == PlotType.Bar:
            figure.subplots_adjust(bottom=0.2)

    return figure


def plot_full_results(
    root_results_directory,
    study_names,
    property_types,
    output_directory,
    dots_per_inch=200,
    sub_plot_size=5.0,
):

    scatter_data = defaultdict(lambda: defaultdict(dict))
    bar_data = defaultdict(lambda: defaultdict(dict))

    for property_type, substance_type in property_types:

        default_unit = property_type.default_unit()
        property_label = property_to_title(property_type, substance_type, default_unit)

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
            estimated_std = data_frame[
                f"Estimated {property_type.__name__} Uncertainty ({default_unit:~})"
            ]

            scatter_data[property_label][study_name] = {
                "main": [estimated_values, estimated_std, measured_values, None]
            }

            rmse_values, _, rmse_ci = compute_bootstrapped_statistics(
                measured_values,
                estimated_values,
                statistics=[Statistics.RMSE],
                bootstrap_iterations=100,
            )

            bar_data["RMSE"][property_label][study_name] = [
                study_index,
                None,
                rmse_values[Statistics.RMSE],
                rmse_ci[Statistics.RMSE],
            ]

    # Plot the scatter data.
    figure = plot_data(
        PlotType.Scatter,
        scatter_data,
        statistics=[Statistics.RMSE, Statistics.R2],
        x_axis_label="Estimated Value",
        y_axis_label="Experimental Value",
        share_x=False,
        dots_per_inch=dots_per_inch,
        sub_plot_size=sub_plot_size,
    )
    figure.savefig(
        os.path.join(output_directory, f"estimated_vs_experiment.pdf"),
        bbox_inches="tight",
    )

    pyplot.close(figure)

    # Plot the RMSE data.
    figure = plot_data(
        PlotType.Bar,
        bar_data,
        share_x=False,
        share_y=False,
        square_axis=False,
        dots_per_inch=dots_per_inch,
        sub_plot_size=sub_plot_size,
    )
    figure.savefig(
        os.path.join(output_directory, f"rmse_per_property.pdf"), bbox_inches="tight",
    )

    pyplot.close(figure)


def plot_per_mixture_type(
    root_results_directory,
    study_names,
    property_types,
    mixture_types,
    training_restrictions,
    output_directory,
    dots_per_inch=200,
    sub_plot_size=5.0,
):

    for training_restriction in training_restrictions:

        scatter_data = defaultdict(lambda: defaultdict(dict))
        bar_data = defaultdict(lambda: defaultdict(dict))

        for property_type, substance_type in property_types:

            default_unit = property_type.default_unit()
            property_label = property_to_title(
                property_type, substance_type, default_unit
            )

            for mixture_type in mixture_types:

                mixture_label = mixture_type.title()

                scatter_data[property_label][mixture_label] = {}
                bar_data[property_label][mixture_type] = {}

                for study_index, study_name in enumerate(study_names):

                    data_frame = load_processed_data_set(
                        os.path.join(
                            root_results_directory,
                            study_name,
                            mixture_type,
                            training_restriction,
                        ),
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
                    estimated_std = data_frame[
                        f"Estimated {property_type.__name__} Uncertainty ({default_unit:~})"
                    ]

                    scatter_data[property_label][mixture_label][study_name] = [
                        estimated_values,
                        estimated_std,
                        measured_values,
                        None,
                    ]

                    rmse_values, _, rmse_ci = compute_bootstrapped_statistics(
                        measured_values,
                        estimated_values,
                        statistics=[Statistics.RMSE],
                        bootstrap_iterations=100,
                    )

                    bar_data[property_label][mixture_type][study_name] = [
                        study_index,
                        None,
                        rmse_values[Statistics.RMSE],
                        rmse_ci[Statistics.RMSE],
                    ]

        # Plot the scatter data.
        figure = plot_data(
            PlotType.Scatter,
            scatter_data,
            statistics=[Statistics.RMSE, Statistics.R2],
            x_axis_label="Estimated Value",
            y_axis_label="Experimental Value",
            share_x=False,
            dots_per_inch=dots_per_inch,
            sub_plot_size=sub_plot_size,
        )
        figure.savefig(
            os.path.join(output_directory, f"{training_restriction}.pdf"),
            bbox_inches="tight",
        )

        pyplot.close(figure)

        # Plot the RMSE data.
        figure = plot_data(
            PlotType.Bar,
            bar_data,
            share_x=False,
            share_y=False,
            square_axis=False,
            dots_per_inch=dots_per_inch,
            sub_plot_size=sub_plot_size,
        )
        figure.savefig(
            os.path.join(output_directory, f"{training_restriction}_rmse.pdf"),
            bbox_inches="tight",
        )

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
    training_restrictions = ["not_in_training", "one_in_training", "both_in_training"]

    root_results_directory = os.path.join("..", "results", "all_results")

    plot_full_results(
        root_results_directory, study_names, property_types, output_directory
    )

    root_results_directory = os.path.join("..", "results", "partitioned_results")

    property_types = [
        (Density, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
    ]

    plot_per_mixture_type(
        root_results_directory,
        study_names,
        property_types,
        mixture_types,
        training_restrictions,
        output_directory,
    )


if __name__ == "__main__":
    main()
