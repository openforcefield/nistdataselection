from enum import Enum

import numpy


class Statistics(Enum):
    Slope = "Slope"
    Intercept = "Intercept"
    R = "R"
    R2 = "R^2"
    P = "p"
    RMSE = "RMSE"
    MSE = "MSE"
    MUE = "MUE"
    Tau = "Tau"


def compute_statistic_unit(base_unit, statistics_type):
    """Computes the correct unit for a given type of statistic.

    Parameters
    ----------
    base_unit: unit.Unit
        The original unit of the property.
    statistics_type: Statistics
        The type of statistic to get the unit for.

    Returns
    -------
    unit.Unit
        The unit the statistic should be given in.
    """
    if statistics_type == Statistics.Slope:
        return None
    elif statistics_type == Statistics.Intercept:
        return base_unit
    elif statistics_type == Statistics.R:
        return None
    elif statistics_type == Statistics.R2:
        return None
    elif statistics_type == Statistics.P:
        return None
    elif statistics_type == Statistics.RMSE:
        return base_unit
    elif statistics_type == Statistics.MSE:
        return base_unit
    elif statistics_type == Statistics.MUE:
        return base_unit
    elif statistics_type == Statistics.Tau:
        return None


def compute_statistics(measured_values, estimated_values, statistics):
    """Calculates a collection of common statistics comparing the measured
    and estimated values.

    Parameters
    ----------
    measured_values: numpy.ndarray
        The experimentally measured values with shape=(number of data points)
    estimated_values: numpy.ndarray
        The computationally estimated values with shape=(number of data points)
    statistics: list of Statistics
        The statistics to compute. If `None`, all statistics will be computed

    Returns
    -------
    numpy.ndarray
        An array of the summarised statistics, containing the
        Slope, Intercept, R, R^2, p, RMSE, MSE, MUE, Tau
    list of str
        Human readable labels for each of the statistics.
    """
    import scipy.stats

    if statistics is None:

        statistics = [
            Statistics.Slope,
            Statistics.Intercept,
            Statistics.R,
            Statistics.R2,
            Statistics.P,
            Statistics.RMSE,
            Statistics.MSE,
            Statistics.MUE,
            Statistics.Tau,
        ]

    summary_statistics = {}

    if (
        len(
            set(statistics).intersection(
                {
                    Statistics.Slope,
                    Statistics.Intercept,
                    Statistics.R,
                    Statistics.R2,
                    Statistics.P,
                }
            )
        )
        > 0
    ):

        (
            summary_statistics[Statistics.Slope],
            summary_statistics[Statistics.Intercept],
            summary_statistics[Statistics.R],
            summary_statistics[Statistics.P],
            _,
        ) = scipy.stats.linregress(measured_values, estimated_values)

        summary_statistics[Statistics.R2] = summary_statistics[Statistics.R] ** 2

    if Statistics.RMSE in statistics:

        summary_statistics[Statistics.RMSE] = numpy.sqrt(
            numpy.mean((estimated_values - measured_values) ** 2)
        )

    if Statistics.MSE in statistics:

        summary_statistics[Statistics.MSE] = numpy.mean(
            estimated_values - measured_values
        )

    if Statistics.MUE in statistics:

        summary_statistics[Statistics.MUE] = numpy.mean(
            numpy.absolute(estimated_values - measured_values)
        )

    if Statistics.Tau in statistics:

        summary_statistics[Statistics.Tau], _ = scipy.stats.kendalltau(
            measured_values, estimated_values
        )

    return numpy.array([summary_statistics[x] for x in statistics]), statistics


def compute_bootstrapped_statistics(
    measured_values,
    measured_stds,
    estimated_values,
    estimated_stds,
    statistics=None,
    percentile=0.95,
    bootstrap_iterations=1000,
):
    """Compute the bootstrapped mean and confidence interval for a set
    of common error statistics.

    Notes
    -----
    Bootstrapped samples are generated with replacement from the full
    original data set.

    Parameters
    ----------
    measured_values: numpy.ndarray
        The experimentally measured values with shape=(n_data_points)
    measured_stds: numpy.ndarray, optional
        The standard deviations in the experimentally measured values with
        shape=(number of data points)
    estimated_values: numpy.ndarray
        The computationally estimated values with shape=(n_data_points)
    estimated_stds: numpy.ndarray, optional
        The standard deviations in the computationally estimated values with
        shape=(number of data points)
    statistics: list of Statistics
        The statistics to compute. If `None`, all statistics will be computed
    percentile: float
        The percentile of the confidence interval to calculate.
    bootstrap_iterations: int
        The number of bootstrap iterations to perform.
    """

    sample_count = len(measured_values)

    # Compute the mean of the statistics.
    mean_statistics, statistics_labels = compute_statistics(
        measured_values, estimated_values, statistics
    )

    # Generate the bootstrapped statistics samples.
    sample_statistics = numpy.zeros((bootstrap_iterations, len(mean_statistics)))

    for sample_index in range(bootstrap_iterations):

        samples_indices = numpy.random.randint(
            low=0, high=sample_count, size=sample_count
        )

        sample_measured_values = measured_values[samples_indices]

        if measured_stds is not None:
            sample_measured_values += numpy.random.normal(0.0, measured_stds)

        sample_estimated_values = estimated_values[samples_indices]

        if estimated_stds is not None:
            sample_estimated_values += numpy.random.normal(0.0, estimated_stds)

        sample_statistics[sample_index], _ = compute_statistics(
            sample_measured_values, sample_estimated_values, statistics
        )

    # Compute the SEM
    standard_errors_array = numpy.std(sample_statistics, axis=0)

    # Store the means and SEMs in dictionaries
    means = dict()
    standard_errors = dict()

    for statistic_index in range(len(mean_statistics)):
        statistic_label = statistics_labels[statistic_index]

        means[statistic_label] = mean_statistics[statistic_index]
        standard_errors[statistic_label] = standard_errors_array[statistic_index]

    # Compute the confidence intervals.
    lower_percentile_index = int(bootstrap_iterations * (1 - percentile) / 2)
    upper_percentile_index = int(bootstrap_iterations * (1 + percentile) / 2)

    confidence_intervals = dict()

    for statistic_index in range(len(mean_statistics)):
        statistic_label = statistics_labels[statistic_index]

        sorted_samples = numpy.sort(sample_statistics[:, statistic_index])

        confidence_intervals[statistic_label] = (
            sorted_samples[lower_percentile_index],
            sorted_samples[upper_percentile_index],
        )

    return means, standard_errors, confidence_intervals
