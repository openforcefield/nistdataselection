import os
from glob import glob

import numpy
import pandas
import seaborn
from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)
from matplotlib import pyplot

from nistdataselection.processing import load_processed_data_set
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import (
    property_to_file_name,
    property_to_title,
    substance_type_to_int,
)
from studies.mixture_feasibility.benchmarking.analysis.alcohol_ester.statistics import (
    Statistics,
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


def plot_bar_with_custom_ci(x, y, lower_bound, upper_bound, **kwargs):

    data = kwargs.pop("data")
    colors = kwargs.pop("color")

    for row_index, (_, row) in enumerate(data.iterrows()):

        ci = numpy.abs([[row[y] - row[lower_bound]], [row[upper_bound] - row[y]]])

        pyplot.bar(
            x=row[x], height=row[y], yerr=ci, label=row[x], color=colors[row_index]
        )


def plot_summary_statistics(
    property_type, statistic_type, output_directory,
):

    summary_data_path = os.path.join("statistics", "per_composition.csv")

    summary_data = pandas.read_csv(summary_data_path)

    property_title = property_to_title(*property_type)

    summary_data = summary_data[summary_data["Property"] == property_title]
    summary_data = summary_data[summary_data["Statistic"] == statistic_type.value]

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


def plot_per_environment_statistics(
    property_types, statistic_types, output_directory,
):

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
        plot.savefig(os.path.join(output_directory, f"{file_name}_per_env.png"))


def plot_estimated_vs_measured(property_types, study_names, output_directory):

    # Refactor the data into a single frame.
    for property_type, substance_type in property_types:

        data_frames = []

        for study_name in study_names:

            results_directory = os.path.join(
                "..", "..", "results", "expanded_set", "partitioned_results", study_name
            )

            environments = [
                os.path.basename(x) for x in glob(os.path.join(results_directory, "*"))
            ]
            environments = [
                x
                for x in environments
                if len(tuple(x.split("_"))) == substance_type_to_int[substance_type]
            ]

            for environment in environments:

                data_frame = load_processed_data_set(
                    os.path.join(results_directory, environment),
                    property_type,
                    substance_type,
                )

                if len(data_frame) == 0:
                    continue

                default_unit = property_type.default_unit()

                measured_values = data_frame[
                    f"Target {property_type.__name__} Value ({default_unit:~})"
                ]
                estimated_values = data_frame[
                    f"Estimated {property_type.__name__} Value ({default_unit:~})"
                ]
                estimated_std = data_frame[
                    f"Estimated {property_type.__name__} Uncertainty ({default_unit:~})"
                ]

                data_frame = pandas.DataFrame()

                data_frame["Measured Value"] = measured_values
                data_frame["Measured Std"] = 0.0

                data_frame["Estimated Value"] = estimated_values
                data_frame["Estimated Std"] = estimated_std

                data_frame["Study"] = study_name
                data_frame["Environment"] = environment

                data_frames.append(data_frame)

        data_frame = pandas.concat(data_frames, ignore_index=True, sort=False)

        plot = seaborn.FacetGrid(
            data_frame,
            col="Study",
            hue="Environment",
            sharex=True,
            sharey=True,
            size=3.5,
            aspect=0.8,
        )
        plot.map(
            pyplot.errorbar,
            "Estimated Value",
            "Measured Value",
            "Measured Std",
            "Estimated Std",
            marker="o",
            linestyle="None",
        )

        plot.set_titles("{col_name}")

        pyplot.subplots_adjust(top=0.8)
        plot.fig.suptitle(property_to_title(property_type, substance_type))

        plot.add_legend()

        file_name = property_to_file_name(property_type, substance_type)
        plot.savefig(os.path.join(output_directory, f"{file_name}.png"))


def main():

    output_directory = "plots"
    os.makedirs(output_directory, exist_ok=True)

    # Define the types of property to plot.
    property_types = [
        # (Density, SubstanceType.Pure),
        # (EnthalpyOfVaporization, SubstanceType.Pure),
        (Density, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
    ]

    # # Define the statistics to plot.
    # statistics = [Statistics.RMSE, Statistics.R2]

    # Plot a summary of each statistic per iteration.
    for property_type in property_types:
        plot_summary_statistics(property_type, Statistics.RMSE, output_directory)

    # # Plot the statistics per environment
    # plot_per_environment_statistics(property_types, statistics, output_directory)


if __name__ == "__main__":
    main()
