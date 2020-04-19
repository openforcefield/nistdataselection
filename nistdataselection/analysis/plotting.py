import os
from glob import glob

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from openforcefield.typing.engines.smirnoff import ForceField

from nistdataselection.processing import load_processed_data_set
from nistdataselection.utils.utils import (
    property_to_file_name,
    property_to_title,
    substance_type_to_int,
)


def plot_categories_with_custom_ci(x, y, hue, lower_bound, upper_bound, **kwargs):

    data = kwargs.pop("data")

    lower_ci = data.pivot(index=x, columns=hue, values=lower_bound)
    upper_ci = data.pivot(index=x, columns=hue, values=upper_bound)

    values = data.pivot(index=x, columns=hue, values=y)

    lower_ci = values - lower_ci
    upper_ci = upper_ci - values

    ci = []

    for column in lower_ci:
        ci.append([lower_ci[column].values, upper_ci[column].values])

    ci = numpy.abs(ci)

    plot_data = data.pivot(index=x, columns=hue, values=y)
    plot_data.plot(kind="bar", yerr=ci, ax=pyplot.gca(), **kwargs)


def plot_categories(x, y, hue, **kwargs):

    data = kwargs.pop("data")

    plot_data = data.pivot(index=x, columns=hue, values=y)
    plot_data.plot(kind="bar", ax=pyplot.gca(), **kwargs)


def plot_bar_with_custom_ci(x, y, lower_bound, upper_bound, **kwargs):

    data = kwargs.pop("data")
    colors = kwargs.pop("color")

    for row_index, (_, row) in enumerate(data.iterrows()):

        ci = numpy.abs([[row[y] - row[lower_bound]], [row[upper_bound] - row[y]]])

        pyplot.bar(
            x=row[x], height=row[y], yerr=ci, label=row[x], color=colors[row_index]
        )


def plot_scatter(x, y, x_err, y_err, hue, hue_order, **kwargs):

    data = kwargs.pop("data")
    colors = kwargs.pop("color")

    for hue_value, color in zip(hue_order, colors):

        hue_data = data[data[hue] == hue_value]

        pyplot.errorbar(
            hue_data[x],
            hue_data[y],
            xerr=hue_data[x_err],
            yerr=hue_data[y_err],
            label=hue_value,
            color=color,
            **kwargs,
        )

    pyplot.gca().plot(
        [0, 1], [0, 1], transform=pyplot.gca().transAxes, color="darkgrey"
    )


def plot_gradient(x, y, hue, hue_order, **kwargs):

    data = kwargs.pop("data")
    colors = kwargs.pop("color")

    for hue_value, color in zip(hue_order, colors):

        hue_data = data[data[hue] == hue_value]

        if len(hue_data) == 0:
            continue

        x_value = hue_data[x].values[0]
        y_value = hue_data[y].values[0]

        norm = numpy.sqrt(x_value * x_value + y_value * y_value)

        x_normalized = x_value / norm
        y_normalized = y_value / norm

        pyplot.plot(
            [0.0, x_normalized], [0.0, y_normalized], color=color, linestyle="--"
        )
        pyplot.plot([0.0, x_value], [0.0, y_value], label=hue_value, color=color)

    axis = pyplot.gca()

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    axis.spines["left"].set_position("center")
    axis.spines["bottom"].set_position("center")

    # Eliminate upper and right axes
    axis.spines["right"].set_color("none")
    axis.spines["top"].set_color("none")

    # Show ticks in the left and lower axes only
    # axis.xaxis.set_ticks_position('bottom')
    # axis.yaxis.set_ticks_position('left')


def plot_estimated_vs_reference(property_types, study_names, output_directory):

    # Refactor the data into a single frame.
    for property_type, substance_type in property_types:

        data_frames = []

        for study_name in study_names:

            results_directory = os.path.join("partitioned_data", study_name)

            environments = [
                os.path.basename(x) for x in glob(os.path.join(results_directory, "*"))
            ]
            environments = [
                x
                for x in environments
                if len(tuple(x.split("_"))) == substance_type_to_int[substance_type]
            ]

            for environment in environments:

                try:

                    data_frame = load_processed_data_set(
                        os.path.join(results_directory, environment),
                        property_type,
                        substance_type,
                    )

                except FileNotFoundError:
                    continue

                if len(data_frame) == 0:
                    continue

                default_unit = property_type.default_unit()

                reference_values = data_frame[
                    f"Reference {property_type.__name__} Value ({default_unit:~})"
                ]
                estimated_values = data_frame[
                    f"Estimated {property_type.__name__} Value ({default_unit:~})"
                ]
                estimated_std = data_frame[
                    f"Estimated {property_type.__name__} Uncertainty ({default_unit:~})"
                ]

                data_frame = pandas.DataFrame()

                data_frame["Reference Value"] = reference_values
                data_frame["Reference Std"] = 0.0

                data_frame["Estimated Value"] = estimated_values
                data_frame["Estimated Std"] = estimated_std

                data_frame["Study"] = study_name
                data_frame["Environment"] = environment

                data_frames.append(data_frame)

        data_frame = pandas.concat(data_frames, ignore_index=True, sort=False)

        environments = list(sorted(set(data_frame["Environment"])))

        palette = seaborn.color_palette("Set1", len(environments))

        plot = seaborn.FacetGrid(
            data_frame,
            col="Study",
            sharex="row",
            sharey="row",
            hue_order=environments,
            palette=palette,
            size=4.0,
            aspect=0.8,
        )
        plot.map_dataframe(
            plot_scatter,
            "Estimated Value",
            "Reference Value",
            "Reference Std",
            "Estimated Std",
            "Environment",
            environments,
            color=palette,
            marker="o",
            linestyle="None",
        )

        plot.set_titles("{col_name}")
        plot.add_legend()

        pyplot.subplots_adjust(top=0.85)

        property_title = property_to_title(property_type, substance_type)
        plot.fig.suptitle(property_title)

        file_name = property_to_file_name(property_type, substance_type)
        plot.savefig(os.path.join(output_directory, f"{file_name}.png"))


def plot_statistic(statistic_types, output_directory):

    statistic_types = [x.value for x in statistic_types]

    summary_data_path = os.path.join("statistics", "all_statistics.csv")

    summary_data = pandas.read_csv(summary_data_path)
    summary_data = summary_data.sort_values("Study")
    summary_data = summary_data[summary_data["Statistic"].isin(statistic_types)]

    study_names = list(sorted({*summary_data["Study"]}))

    palette = seaborn.color_palette(n_colors=len(study_names))

    plot = seaborn.FacetGrid(
        summary_data,
        col="Property",
        row="Statistic",
        size=4.0,
        aspect=1.0,
        sharey=False,
        # margin_titles=True
        # gridspec_kws={"wspace": 0.2},
    )
    plot.map_dataframe(
        plot_bar_with_custom_ci,
        f"Study",
        f"Value",
        f"Lower 95% CI",
        f"Upper 95% CI",
        color=palette,
    )

    plot.set_titles("{row_name}|{col_name}")

    for i, axes_row in enumerate(plot.axes):

        for j, axes_col in enumerate(axes_row):

            row, col = axes_col.get_title().split("|")

            if i == 0:
                axes_col.set_title(col.strip())
            else:
                axes_col.set_title("")

            if j == 0:
                axes_col.set_ylabel(f"${row.strip()}$")

            axes_col.set_xticklabels([])

    plot.add_legend()

    plot.savefig(os.path.join(output_directory, "statistics.png"))


def plot_statistic_per_environment(
    property_types, statistic_types, output_directory, per_composition=False
):

    if per_composition:
        per_environment_data_path = os.path.join("statistics", "per_composition.csv")
    else:
        per_environment_data_path = os.path.join("statistics", "per_environment.csv")

    per_environment_data = pandas.read_csv(per_environment_data_path)

    statistic_types = [x.value for x in statistic_types]

    study_names = list(sorted({*per_environment_data["Study"]}))

    for property_type, substance_type in property_types:

        property_title = property_to_title(property_type, substance_type)

        property_data = per_environment_data[
            per_environment_data["Property"] == property_title
        ]
        property_data = property_data[property_data["Statistic"].isin(statistic_types)]

        palette = seaborn.color_palette(n_colors=len(study_names))

        plot = seaborn.FacetGrid(
            property_data,
            col="Property",
            row="Statistic",
            height=4.0,
            aspect=4.0,
            sharey=False,
        )
        plot.map_dataframe(
            plot_categories_with_custom_ci,
            "Environment",
            f"Value",
            "Study",
            f"Lower 95% CI",
            f"Upper 95% CI",
            color=palette,
        )

        plot.add_legend()

        plot.set_titles("{row_name}|{col_name}")

        for i, axes_row in enumerate(plot.axes):

            for j, axes_col in enumerate(axes_row):

                row, col = axes_col.get_title().split("|")

                if i == 0:
                    axes_col.set_title(col.strip())
                else:
                    axes_col.set_title("")

                if j == 0:
                    axes_col.set_ylabel(f"${row.strip()}$")

        file_name = property_to_file_name(property_type, substance_type)

        plot.savefig(
            os.path.join(output_directory, f"{file_name}_statistics_per_env.png")
        )


def plot_statistic_per_iteration(
    property_type, statistic_type, output_directory, per_composition=False
):

    data_name = "per_environment.csv" if not per_composition else "per_composition.csv"

    summary_data_path = os.path.join("statistics", data_name)

    summary_data = pandas.read_csv(summary_data_path)

    property_title = property_to_title(*property_type)

    summary_data = summary_data[summary_data["Property"] == property_title]
    summary_data = summary_data[summary_data["Statistic"] == statistic_type.value]

    if len(summary_data) == 0:
        return

    summary_data_iter_0 = summary_data[summary_data["Iteration"] == 0]
    summary_data_iter_remaining = summary_data[summary_data["Iteration"] > 0]

    summary_data_iter_remaining = pandas.merge(
        summary_data_iter_remaining,
        summary_data_iter_0,
        on=["Study", "Property", "Statistic", "Environment"],
        suffixes=("", "_0"),
    )

    summary_data_iter_remaining["Value"] = (
        summary_data_iter_remaining["Value"] - summary_data_iter_remaining["Value_0"]
    )
    summary_data_iter_remaining["Lower 95% CI"] = (
        summary_data_iter_remaining["Lower 95% CI"]
        - summary_data_iter_remaining["Value_0"]
    )
    summary_data_iter_remaining["Upper 95% CI"] = (
        summary_data_iter_remaining["Upper 95% CI"]
        - summary_data_iter_remaining["Value_0"]
    )

    environments = list(sorted({*summary_data["Environment"]}))

    palette = seaborn.color_palette(n_colors=len(environments))

    plot = seaborn.FacetGrid(
        summary_data_iter_remaining,
        col="Statistic",
        row="Study",
        size=4.0,
        aspect=10.0,
        sharey=False,
    )
    plot.map_dataframe(
        plot_categories_with_custom_ci,
        "Iteration",
        "Value",
        "Environment",
        "Lower 95% CI",
        "Upper 95% CI",
        color=palette,
    )

    plot.set_titles("{col_name}|{row_name}")

    for i, axes_row in enumerate(plot.axes):

        for j, axes_col in enumerate(axes_row):

            row, col = axes_col.get_title().split("|")

            axes_col.set_title(col.strip())

            if j == 0:
                axes_col.set_ylabel(f"{row.strip()}")

    plot.add_legend()

    file_name = property_to_file_name(*property_type)
    file_name = f"{file_name}_{statistic_type.value.lower()}.png"

    plot.savefig(os.path.join(output_directory, file_name))


def plot_objective_per_iteration(study_names, output_directory):

    objective_function_per_study = []

    for study_name in study_names:

        objective_function = pandas.read_csv(
            os.path.join("all_data", study_name, "objective_function.csv")
        )

        objective_function["Study"] = study_name

        objective_function_per_study.append(objective_function)

    objective_function_per_study = pandas.concat(
        objective_function_per_study, ignore_index=True, sort=False
    )

    plot = seaborn.FacetGrid(
        objective_function_per_study,
        col="Study",
        size=4.0,
        aspect=1.0,
        sharex=False,
        sharey=False,
    )
    plot.map(
        pyplot.plot, "Iteration", "Objective Function",
    )

    plot.set_titles("{col_name}")

    plot.savefig(os.path.join(output_directory, "objective_function.png"))


def plot_gradient_per_environment(
    property_types, study_names, output_directory, iteration
):

    gradient_data = pandas.read_csv(os.path.join("gradients", "per_composition.csv"))
    gradient_data = gradient_data[gradient_data["Iteration"] == iteration]

    # Refactor the data into a single frame.
    for property_type, substance_type in property_types:

        property_title = property_to_title(property_type, substance_type)

        for study_name in study_names:

            study_data = gradient_data[
                (gradient_data["Study"] == study_name)
                & (gradient_data["Property"] == property_title)
            ]

            if len(study_data) == 0:
                continue

            environments = list(sorted(set(study_data["Environment"])))

            palette = seaborn.color_palette("Set1", len(environments))

            plot = seaborn.FacetGrid(
                study_data,
                col="Smirks",
                sharex=False,
                sharey=False,
                hue_order=environments,
                palette=palette,
                size=4.0,
                aspect=1.0,
            )
            plot.map_dataframe(
                plot_gradient,
                "epsilon",
                "rmin_half",
                "Environment",
                environments,
                color=palette,
                marker="o",
                linestyle="None",
            )

            plot.set_titles("{col_name}")
            plot.add_legend()

            max_axis_lim = -1e10

            for _, axes_row in enumerate(plot.axes):
                for _, axis in enumerate(axes_row):

                    x_lim = tuple(abs(x) for x in axis.get_xlim())
                    y_lim = tuple(abs(x) for x in axis.get_ylim())

                    max_axis_lim = max([max_axis_lim, *x_lim, max(y_lim)])

            for _, axes_row in enumerate(plot.axes):
                for _, axis in enumerate(axes_row):
                    axis.set_xlim((-max_axis_lim, max_axis_lim))
                    axis.set_ylim((-max_axis_lim, max_axis_lim))

                    axis.xaxis.labelpad = 120
                    axis.yaxis.labelpad = 120

            pyplot.subplots_adjust(top=0.85)

            property_title = property_to_title(property_type, substance_type)
            plot.fig.suptitle(property_title)

            file_name = property_to_file_name(property_type, substance_type)
            file_name = f"{file_name}_iter_{iteration}"

            folder_path = os.path.join(output_directory, "gradients", study_name)
            os.makedirs(folder_path, exist_ok=True)

            plot.savefig(os.path.join(folder_path, f"{file_name}_gradient.png"))

            pyplot.close("all")


def plot_parameter_changes(
    original_parameter_path,
    optimized_parameter_directory,
    study_names,
    parameter_smirks,
    output_directory,
):

    from simtk import unit as simtk_unit

    parameter_attributes = ["epsilon", "rmin_half"]
    default_units = {
        "epsilon": simtk_unit.kilocalories_per_mole,
        "rmin_half": simtk_unit.angstrom,
    }

    # Find the values of the original and optimized parameters.
    data_rows = []

    for study_name in study_names:

        original_force_field = ForceField(
            original_parameter_path, allow_cosmetic_attributes=True,
        )
        optimized_force_field = ForceField(
            os.path.join(optimized_parameter_directory, f"{study_name}.offxml"),
            allow_cosmetic_attributes=True,
        )

        original_handler = original_force_field.get_parameter_handler("vdW")
        optimized_handler = optimized_force_field.get_parameter_handler("vdW")

        for parameter in original_handler.parameters:

            if parameter.smirks not in parameter_smirks:
                continue

            for attribute_type in parameter_attributes:

                original_value = getattr(parameter, attribute_type)
                optimized_value = getattr(
                    optimized_handler.parameters[parameter.smirks], attribute_type
                )

                percentage_change = optimized_value - original_value

                data_row = {
                    "Study": study_name,
                    "Smirks": parameter.smirks,
                    "Attribute": f"{attribute_type} ({default_units[attribute_type]})",
                    "Delta": percentage_change.value_in_unit(
                        default_units[attribute_type]
                    ),
                }

                data_rows.append(data_row)

    parameter_data = pandas.DataFrame(data_rows)

    palette = seaborn.color_palette(n_colors=len(study_names))

    plot = seaborn.FacetGrid(parameter_data, row="Attribute", height=4.0, aspect=2.0,)
    plot.map_dataframe(plot_categories, "Smirks", "Delta", "Study", color=palette)

    plot.add_legend()

    plot.savefig(os.path.join(output_directory, f"parameter_changes.png"))
