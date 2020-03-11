"""A script to build the fitting options for the simulation / reweighting
comparison study.
"""
import argparse
import json
import os
from collections import defaultdict
from glob import glob

import numpy
import pandas
from evaluator.datasets import PhysicalPropertyDataSet
from matplotlib import pyplot

from nistdataselection.utils.utils import property_to_snake_case


def plot_average_contribution_per_property(
    training_data_set, statistics_directory, output_directory
):
    """Plots the average contribution of each property to the
    objective function at each iteration.

    Parameters
    ----------
    training_data_set: PhysicalPropertyDataSet
        The data set which was optimized against.
    statistics_directory: str
        The directory which contains extracted fitting statistics.
    output_directory: str
        The directory to store the plots in.
    """

    # Determine how many iterations ForceBalance has completed.
    n_iterations = len(glob(f"{statistics_directory}/iter*")) / len(
        training_data_set.property_types
    )

    contributions_per_property = defaultdict(list)
    contribution_std_per_property = defaultdict(list)

    for iteration_index in range(int(n_iterations)):

        iteration = "iter_" + str(iteration_index).zfill(4)

        for property_type in training_data_set.property_types:

            property_type = property_to_snake_case(property_type)

            file_name = os.path.join(
                statistics_directory, f"{iteration}_{property_type}.csv"
            )

            statistics = pandas.read_csv(file_name)

            average_contribution = numpy.mean(statistics["Term"])
            std_contribution = numpy.std(statistics["Term"])

            contributions_per_property[property_type].append(average_contribution)
            contribution_std_per_property[property_type].append(std_contribution)

    figure, axis = pyplot.subplots(1, 1, figsize=(4, 4))

    axis.set_xlabel("Iteration")
    axis.set_ylabel("Avg. Objective Function")

    for property_type in contributions_per_property:

        axis.errorbar(
            range(len(contributions_per_property[property_type])),
            contributions_per_property[property_type],
            yerr=contribution_std_per_property[property_type],
            marker="o",
            label=property_type,
        )

    axis.legend(
        bbox_to_anchor=(0.0, -0.275, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=2,
    )

    figure.tight_layout()
    figure.savefig(os.path.join(output_directory, "avg_contribution.pdf"))
    figure.savefig(os.path.join(output_directory, "avg_contribution.png"))


def plot_object_function(statistics_directory, output_directory):
    """Plots the average contribution of each property to the
    objective function at each iteration.

    Parameters
    ----------
    statistics_directory: str
        The directory which contains extracted fitting statistics.
    output_directory: str
        The directory to store the plots in.
    """

    with open(os.path.join(statistics_directory, "objective_function.json")) as file:
        objective_function = json.load(file)

    figure, axis = pyplot.subplots(1, 1, figsize=(4, 4))

    axis.set_xlabel("Iteration")
    axis.set_ylabel("Objective Function")

    axis.plot(range(len(objective_function)), objective_function, marker="o")

    figure.tight_layout()
    figure.savefig(os.path.join(output_directory, "objective_function.pdf"))
    figure.savefig(os.path.join(output_directory, "objective_function.png"))


def main(data_set_path, statistics_directory, output_directory):

    os.makedirs(output_directory, exist_ok=True)

    target_data_set = PhysicalPropertyDataSet.from_json(data_set_path)

    plot_average_contribution_per_property(
        target_data_set, statistics_directory, output_directory
    )

    plot_object_function(statistics_directory, output_directory)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plots different statistics about the optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        "-dat",
        type=str,
        help="The file path to the training data set.",
        required=True,
    )
    parser.add_argument(
        "--statistics",
        "-s",
        type=str,
        help="The path to the directory which contains the extracted fitting "
        "statistics.",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="The directory to store the output plots in.",
        required=False,
        default="plots",
    )

    args = parser.parse_args()
    main(args.dataset, args.statistics, args.output)
