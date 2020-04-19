import functools
import itertools
import logging
import os
from collections import defaultdict
from glob import glob
from multiprocessing import Pool

import numpy
import pandas
from evaluator import unit
from evaluator.attributes import UNDEFINED
from evaluator.client import RequestResult
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.substances import MoleFraction
from forcebalance.evaluator_io import Evaluator_SMIRNOFF
from forcebalance.nifty import lp_load
from scipy.optimize import linear_sum_assignment

from nistdataselection.analysis.statistics import compute_statistics
from nistdataselection.curation.filtering import filter_by_checkmol
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils.utils import (
    SubstanceType,
    analyse_functional_groups,
    int_to_substance_type,
    property_to_title,
)

logger = logging.getLogger(__name__)


def _partition_by_environment(data_per_property, environments):
    """Partition a set of results into results collected for specific
    chemical environments (e.g. alcohol-ester mixtures)

    Parameters
    ----------
    data_per_property: dict of tuple of pandas.DataFrame
        The full set of data for each type of property.
    environments: dict of str and list of str
        The chemical environments of interest.

    Returns
    -------
    dict of tuple of str and dict of tuple of pandas.DataFrame
        The partitioned data.
    """

    environment_pairs = [(x, x) for x in environments]
    # noinspection PyTypeChecker
    environment_pairs.extend(itertools.combinations(environments, 2))

    substance_type_environments = {
        SubstanceType.Pure: [(x,) for x in environments],
        SubstanceType.Binary: environment_pairs,
    }

    data_per_environment = defaultdict(dict)

    for property_tuple, data_frame in data_per_property.items():

        _, substance_type = property_tuple

        for environment_types in substance_type_environments[substance_type]:

            chemical_environments = [environments[x] for x in environment_types]

            environment_data_frame = filter_by_checkmol(
                data_frame, *chemical_environments
            )
            environment_data_frame = environment_data_frame.copy()

            if len(environment_data_frame) == 0:
                continue

            for index in range(environment_data_frame["N Components"].max()):
                environment_data_frame[f"Environment {index + 1}"] = None

            for row_index, data_row in environment_data_frame.iterrows():

                n_components = data_row["N Components"]

                component_smiles = [
                    data_row[f"Component {index + 1}"] for index in range(n_components)
                ]
                component_moieties = [
                    analyse_functional_groups(x) for x in component_smiles
                ]

                matches_grid = []

                for moieties in component_moieties:

                    matches_row = []

                    for environment_type in environment_types:

                        intersection = set.intersection(
                            {*moieties}, {*environments[environment_type]}
                        )

                        count = sum(moieties[x] for x in intersection)
                        matches_row.append(count)

                    matches_grid.append(matches_row)

                matches_grid = numpy.array(matches_grid)

                smiles_indices, environment_indices = linear_sum_assignment(
                    matches_grid, maximize=True
                )

                for smiles_index, environment_index in zip(
                    smiles_indices, environment_indices
                ):

                    environment_data_frame.loc[
                        row_index, f"Environment {smiles_index + 1}"
                    ] = environment_types[environment_index]

            data_per_environment[environment_types][
                property_tuple
            ] = environment_data_frame

    return data_per_environment


def _partition_by_composition(data_frame, environment):
    """Splits a data frame data based on the composition of
    the different environments in a substance.

    E.g pure, <50% env 1 and >50% env 2, 50% env 1 and 50% env 2,
    >50% env 1 and <50% env 2

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data to partition.
    environment: str
        The environment present in the data frame.

    Returns
    -------
    dict of str and pandas.DataFrame
        The partitioned data.
    """

    if data_frame["N Components"].max() < 2:
        return {environment: data_frame}

    n_components = data_frame["N Components"].max()

    environments = set()

    for index in range(n_components):
        environments.update(data_frame[f"Environment {index + 1}"].values)

    environments = list(sorted(environments))

    if len(environments) == 1 and environment.split("_")[0] == environments[0]:
        return {environment: data_frame}

    if len(environments) != 2:
        raise NotImplementedError()

    env_1_excess = data_frame[
        (
            (data_frame["Environment 1"] == environments[0])
            & (data_frame["Mole Fraction 1"] > 0.55)
        )
        | (
            (data_frame["Environment 2"] == environments[0])
            & (data_frame["Mole Fraction 2"] > 0.55)
        )
    ]
    env_2_excess = data_frame[
        (
            (data_frame["Environment 1"] == environments[0])
            & (data_frame["Mole Fraction 1"] < 0.45)
        )
        | (
            (data_frame["Environment 2"] == environments[0])
            & (data_frame["Mole Fraction 2"] < 0.45)
        )
    ]

    remaining_ids = {*data_frame["Id"]}
    remaining_ids -= {*env_1_excess["Id"]}
    remaining_ids -= {*env_2_excess["Id"]}

    roughly_equal = data_frame[data_frame["Id"].isin(remaining_ids)]

    return {
        f"{environments[0]} > {environments[1]}": env_1_excess,
        f"{environments[0]} ~ {environments[1]}": roughly_equal,
        f"{environments[0]} < {environments[1]}": env_2_excess,
    }


def compute_objective_statistics(
    data_per_property, estimated_data_set, target_options=None
):
    """Computes information about each pair of reference and estimated
    properties which was used in a ForceBalance optimization, such as
    each propertied contribution to the object function, and the gradient
    of such with respect to each of the re-fitted parameters.

    Parameters
    ----------
    data_per_property: dict of tuple and pandas.DataFrame
        Data frames containing the sets of reference and estimated
        properties.
    estimated_data_set: PhysicalPropertyDataSet
        The estimated properties which contain the property gradients.
    target_options: Evaluator_SMIRNOFF.OptionsFile, optional
        The options used during the optimization.

    Returns
    -------
    dict of tuple and pandas.DataFrame
        The data frames containing the extra fitting
        statistics.
    """

    estimated_properties_by_id = {x.id: x for x in estimated_data_set}

    n_properties_by_type = defaultdict(int)

    for (property_type, _), data_frame in data_per_property.items():
        n_properties_by_type[property_type.__name__] += len(data_frame)

    for property_tuple in data_per_property:

        data_frame = data_per_property[property_tuple]

        property_type = property_tuple[0].__name__

        default_unit = property_tuple[0].default_unit()
        target_unit = target_options.denominators[property_type].units

        reference_value_header = f"Reference {property_type} Value ({default_unit:~})"
        estimated_value_header = f"Estimated {property_type} Value ({default_unit:~})"

        unit_scale = (1.0 * default_unit).to(target_unit).magnitude

        delta = data_frame[reference_value_header] - data_frame[estimated_value_header]
        delta *= unit_scale

        delta_sqr = delta ** 2

        normalised_weight = (
            target_options.weights[property_type] / n_properties_by_type[property_type]
        )
        denominator = (
            target_options.denominators[property_type].to(target_unit).magnitude
        )

        objective_contribution = normalised_weight * (delta / denominator) ** 2

        data_frame["Delta^2"] = delta_sqr
        data_frame["Weight"] = normalised_weight
        data_frame["Denom"] = denominator
        data_frame["Term"] = objective_contribution

        gradient_rows = []

        for property_id in data_frame["Id"]:

            estimated_property = estimated_properties_by_id[property_id]
            gradient_row = {"Id": property_id}

            property_data = data_frame[data_frame["Id"] == property_id]

            for gradient in estimated_property.gradients:

                gradient_unit = target_unit

                if gradient.key.attribute == "epsilon":
                    gradient_unit /= unit.kilocalorie / unit.mole
                elif gradient.key.attribute == "rmin_half":
                    gradient_unit /= unit.angstrom

                gradient_value = gradient.value.to(gradient_unit).magnitude

                delta_value = (
                    property_data[reference_value_header]
                    - property_data[estimated_value_header]
                )
                delta_value = float(delta_value.values[0]) * unit_scale

                objective_gradient = (
                    2.0
                    * normalised_weight
                    * delta_value
                    * gradient_value
                    / denominator ** 2
                )
                header = f"d Term / d {gradient.key.smirks} {gradient.key.attribute}"

                gradient_row[header] = objective_gradient

            gradient_rows.append(gradient_row)

        gradient_frame = pandas.DataFrame(gradient_rows)

        data_frame = pandas.merge(data_frame, gradient_frame, on="Id")
        data_per_property[property_tuple] = data_frame


def combine_data_sets(reference_data_set, estimated_data_set):
    """Combines a data set of reference properties, and a data set
    of those properties estimated from simulation / simulation
    data into a single data frame per type of property.

    Parameters
    ----------
    reference_data_set: PhysicalPropertyDataSet
        The reference data set which was optimized against.
    estimated_data_set: PhysicalPropertyDataSet
        The reference data evaluated using simulations.

    Returns
    -------
    dict of tuple and pandas.DataFrame
        The combined data frames per each type of property.
    """
    reference_properties_by_id = {x.id: x for x in reference_data_set}
    estimated_properties_by_id = {x.id: x for x in estimated_data_set}

    property_ids = [*reference_properties_by_id.keys()]

    data_per_property = defaultdict(list)

    for property_id in property_ids:

        # Make sure the reference property was actually estimated.
        if property_id not in estimated_properties_by_id:

            logger.warning(
                f"The reference property with id={property_id} was not found in the"
                f"estimated data set"
            )

            continue

        reference_property = reference_properties_by_id[property_id]
        estimated_property = estimated_properties_by_id[property_id]

        temperature = reference_property.thermodynamic_state.temperature.to(
            unit.kelvin
        ).magnitude
        pressure = reference_property.thermodynamic_state.pressure.to(
            unit.kilopascal
        ).magnitude

        components = []
        mole_fractions = []

        for component in reference_property.substance:

            for x in reference_property.substance.get_amounts(component):

                if not isinstance(x, MoleFraction):
                    continue

                mole_fractions.append(x.value)
                break

            components.append(component.smiles)

        property_type = reference_property.__class__
        default_unit = property_type.default_unit()

        reference_value = reference_property.value.to(default_unit).magnitude
        reference_std = (
            numpy.nan
            if reference_property.uncertainty is UNDEFINED
            else reference_property.uncertainty.to(default_unit).magnitude
        )

        estimated_value = estimated_property.value.to(default_unit).magnitude
        estimated_std = (
            numpy.nan
            if estimated_property.uncertainty is UNDEFINED
            else estimated_property.uncertainty.to(default_unit).magnitude
        )

        data_row = {
            "Id": property_id,
            "Temperature (K)": temperature,
            "Pressure (kPa)": pressure,
            "N Components": len(reference_property.substance),
        }

        for index in range(len(components)):
            data_row[f"Component {index + 1}"] = components[index]
            data_row[f"Mole Fraction {index + 1}"] = mole_fractions[index]

        data_row[
            f"Reference {property_type.__name__} Value ({default_unit:~})"
        ] = reference_value
        data_row[
            f"Reference {property_type.__name__} Uncertainty ({default_unit:~})"
        ] = reference_std

        data_row[
            f"Estimated {property_type.__name__} Value ({default_unit:~})"
        ] = estimated_value
        data_row[
            f"Estimated {property_type.__name__} Uncertainty ({default_unit:~})"
        ] = estimated_std

        substance_type = int_to_substance_type[len(reference_property.substance)]
        data_per_property[(property_type, substance_type)].append(data_row)

    data_frames_per_property = {
        x: pandas.DataFrame(y) for x, y in data_per_property.items()
    }

    return data_frames_per_property


def combine_results(reference_data_set, estimated_data_set, target_options=None):
    """Combines a data set of reference properties, and a data set
    of those properties estimated from simulation / simulation
    data into a single data frame per type of property. If a set of
    `target_options` are provided, information used by ForceBalance
    will also be included in the final data frame.

    Parameters
    ----------
    reference_data_set: PhysicalPropertyDataSet
        The reference data set.
    estimated_data_set: PhysicalPropertyDataSet
        The reference data evaluated using simulations.
    target_options: Evaluator_SMIRNOFF.OptionsFile, optional
        The options used during the optimization.

    Returns
    -------
    dict of tuple and pandas.DataFrame
        The statistics per type of property
    """

    data_per_property = combine_data_sets(
        reference_data_set=reference_data_set, estimated_data_set=estimated_data_set
    )

    if target_options is not None:

        compute_objective_statistics(
            data_per_property, estimated_data_set, target_options
        )

    return data_per_property


def processes_optimization_results(
    study_name, study_path, environments, root_output_directory
):
    """Extracts the results of a ForceBalance optimization

    Parameters
    ----------
    study_name: str
        The name assigned to the optimization.
    study_path
        The path to the root optimization directory (i.e. the
        directory which contains the targets directory).
    environments: dict of str and list of str
        The chemical environments which were included in the
        optimization.
    root_output_directory: str
        The root directory to save the processed results in.
    """

    root_target_directory = os.path.join(study_path, "targets")

    # Find the target options and training set files.
    target_names = [x for x in os.listdir(root_target_directory)]
    assert len(target_names) == 1

    target_name = os.path.basename(target_names[0])

    target_directory = os.path.join(root_target_directory, target_name)

    target_data_set = PhysicalPropertyDataSet.from_json(
        os.path.join(target_directory, "training_set.json")
    )
    target_options = Evaluator_SMIRNOFF.OptionsFile.from_json(
        os.path.join(target_directory, "options.json")
    )

    # Determine the number of optimization iterations.
    results_directory = os.path.join(study_path, "optimize.tmp", target_name)
    n_iterations = len(glob(os.path.join(results_directory, "iter_*")))

    # Extract data about each iteration.
    objective_function_data = []

    for iteration in range(n_iterations):

        folder_name = "iter_" + str(iteration).zfill(4)
        results_path = os.path.join(results_directory, folder_name, "results.json")

        if not os.path.isfile(results_path):
            continue

        results = RequestResult.from_json(results_path)
        estimated_properties = results.estimated_properties

        results_per_property = combine_results(
            target_data_set, estimated_properties, target_options
        )

        # Save the results
        output_directory = os.path.join("all_data", study_name, folder_name)

        for property_type, results_frame in results_per_property.items():
            save_processed_data_set(output_directory, results_frame, *property_type)

        # Partition the results per environment
        results_per_environment = _partition_by_environment(
            results_per_property, environments
        )

        for environment_types in results_per_environment:

            output_directory = os.path.join(
                root_output_directory,
                "partitioned_data",
                study_name,
                "_".join(environment_types),
                folder_name,
            )

            for property_type in results_per_environment[environment_types]:

                results_frame = results_per_environment[environment_types][
                    property_type
                ]

                save_processed_data_set(output_directory, results_frame, *property_type)

        # Extract the value of this iterations objective function
        file_path = os.path.join(results_directory, folder_name, "objective.p")

        statistics = lp_load(file_path)

        objective_function_data.append(
            {"Iteration": iteration, "Objective Function": statistics["X"]}
        )

    output_directory = os.path.join(root_output_directory, "all_data", study_name)

    objective_data_frame = pandas.DataFrame(objective_function_data)
    objective_data_frame.to_csv(
        os.path.join(output_directory, "objective_function.csv")
    )


def processes_benchmark_results(
    study_name, results_path, reference_path, environments, root_output_directory
):
    """Extracts the results of an OpenFF Evaluator benchmark.

    Parameters
    ----------
    study_name: str
        The name assigned to the benchmark.
    reference_path: str
        The path to the reference property data set.
    results_path: str
        The path to the request results object.
    environments: dict of str and list of str
        The chemical environments which were included in the
        benchmark.
    root_output_directory: str
        The root directory to save the processed results in.
    """

    reference_data_set = PhysicalPropertyDataSet.from_json(reference_path)

    results = RequestResult.from_json(results_path)
    estimated_data_set = results.estimated_properties

    results_per_property = combine_results(reference_data_set, estimated_data_set)

    # Save the results
    output_directory = os.path.join("all_data", study_name)

    for property_type, results_frame in results_per_property.items():
        save_processed_data_set(output_directory, results_frame, *property_type)

    # Partition the results per environment
    results_per_environment = _partition_by_environment(
        results_per_property, environments
    )

    for environment_types in results_per_environment:

        output_directory = os.path.join(
            root_output_directory,
            "partitioned_data",
            study_name,
            "_".join(environment_types),
        )

        for property_type in results_per_environment[environment_types]:

            results_frame = results_per_environment[environment_types][property_type]

            save_processed_data_set(output_directory, results_frame, *property_type)


def _compute_data_row(
    data_frame,
    property_type,
    substance_type,
    study_name,
    environment,
    iteration,
    partition_by_composition,
    bootstrap_iterations,
):

    if len(data_frame) == 0:
        return []

    data_frames = {environment: data_frame}

    if partition_by_composition:
        data_frames = _partition_by_composition(data_frame, environment)

    data_rows = []

    for environment_label, data_frame in data_frames.items():

        data_frame = data_frame.copy()
        data_frame.reset_index(drop=True, inplace=True)

        statistics, statistics_std, statistics_ci = compute_statistics(
            data_frame, property_type, bootstrap_iterations
        )

        property_title = property_to_title(property_type, substance_type)

        for statistic in statistics:

            data_row = {
                "Study": study_name,
                "Property": property_title,
                "Statistic": statistic.value,
                "Value": statistics[statistic],
                "Std": statistics_std[statistic],
                "Lower 95% CI": statistics_ci[statistic][0],
                "Upper 95% CI": statistics_ci[statistic][1],
            }

            if environment is not None:
                data_row["Environment"] = environment_label

            if iteration is not None:
                data_row["Iteration"] = iteration

            data_rows.append(data_row)

    return data_rows


def _compute_data_rows(
    property_type,
    study_name,
    environment,
    root_data_directory,
    partition_by_composition,
    bootstrap_iterations,
):

    if partition_by_composition and environment is None:

        raise ValueError(
            "The data can only be partioned by composition when an environment is "
            "specified."
        )

    property_type, substance_type = property_type

    data_path = os.path.join(root_data_directory, study_name)

    if environment is not None:

        environment = "_".join(environment)
        data_path = os.path.join(data_path, environment)

    n_iterations = len(glob(os.path.join(data_path, "iter_*")))

    data_rows = []

    if n_iterations == 0:

        try:

            data_frame = load_processed_data_set(
                data_path, property_type, substance_type
            )

        except FileNotFoundError:
            return []

        data_rows.extend(
            _compute_data_row(
                data_frame,
                property_type,
                substance_type,
                study_name,
                environment,
                None,
                partition_by_composition,
                bootstrap_iterations,
            )
        )

    for iteration in range(n_iterations):

        iteration_name = "iter_" + str(iteration).zfill(4)
        iteration_data_path = os.path.join(data_path, iteration_name)

        try:

            data_frame = load_processed_data_set(
                iteration_data_path, property_type, substance_type
            )

        except FileNotFoundError:
            continue

        data_rows.extend(
            _compute_data_row(
                data_frame,
                property_type,
                substance_type,
                study_name,
                environment,
                iteration,
                partition_by_composition,
                bootstrap_iterations,
            )
        )

    return data_rows


def generate_statistics(
    root_data_directory,
    property_types,
    study_names,
    environments,
    partition_by_composition,
    bootstrap_iterations,
    n_processes=1,
):
    """Computes various statistics collected during either a benchmark or
    optimization study, such as the RMSE and R2.

    Parameters
    ----------
    root_data_directory: str
        The path to the directory containing the data. It is expected
        that this directory was created using either

            * `processes_optimization_results`
            * `processes_benchmark_results`

    property_types: list of tuple of type of PhysicalProperty and SubstanceType
        The properties to compute statistics for.
    study_names: list of str
        The names of the studies to compute statistics for.
    environments: dict of str and list of str, optional
        The chemical environments to compute statistics for.
    partition_by_composition: bool
        Whether to partition the different environments specified by
        `environments` further based on the composition of the individual
        environments any mixtures.
    bootstrap_iterations: int
        The number of bootstrapping iterations to perform when computing any
        statistics.
    n_processes: int
        The number of processes to parallelize the computations over.

    Returns
    -------

    """

    inputs = []

    if environments is None:
        environments = {SubstanceType.Pure: [None], SubstanceType.Binary: [None]}

    for property_type in property_types:
        for study_name in study_names:
            for environment in environments[property_type[1]]:
                inputs.append((property_type, study_name, environment))

    with Pool(n_processes) as pool:

        data_rows = list(
            pool.starmap(
                functools.partial(
                    _compute_data_rows,
                    partition_by_composition=partition_by_composition,
                    root_data_directory=root_data_directory,
                    bootstrap_iterations=bootstrap_iterations,
                ),
                inputs,
            )
        )

    data_rows = [x for x in data_rows if x is not None and len(x) > 0]
    data_rows = [y for x in data_rows for y in x]

    data_frame = pandas.DataFrame(data_rows)

    return data_frame


def extract_gradients(
    root_data_directory,
    property_types,
    study_names,
    environments,
    partition_by_composition,
):
    """Computes the average gradient of each property with respect
    to the fitted parameters for each environment of interest.

    Parameters
    ----------
    root_data_directory: str
        The path to the directory containing the data. It is expected
        that this directory was created using either

            * `processes_optimization_results`
            * `processes_benchmark_results`

    property_types: list of tuple of type of PhysicalProperty and SubstanceType
        The properties to compute the gradients of.
    study_names: list of str
        The names of the studies to compute gradients for.
    environments: dict of str and list of str, optional
        The chemical environments to compute gradients for.
    partition_by_composition: bool
        Whether to partition the different environments specified by
        `environments` further based on the composition of the individual
        environments any mixtures.
    """

    inputs = []

    if environments is None:
        environments = {SubstanceType.Pure: [None], SubstanceType.Binary: [None]}

    # Pre-compute the triple loop to avoid overly indented code.
    for property_type in property_types:
        for study_name in study_names:
            for environment in environments[property_type[1]]:
                inputs.append((property_type, study_name, environment))

    data_rows = []

    for property_type, study_name, environment in inputs:

        if partition_by_composition and environment is None:

            raise ValueError(
                "The data can only be partioned by composition when an environment is "
                "specified."
            )

        property_type, substance_type = property_type
        property_title = property_to_title(property_type, substance_type)

        # Define the root path to the data.
        data_path = os.path.join(root_data_directory, study_name)

        if environment is not None:

            environment = "_".join(environment)
            data_path = os.path.join(data_path, environment)

        n_iterations = len(glob(os.path.join(data_path, "iter_*")))

        for iteration in range(n_iterations):

            iteration_name = "iter_" + str(iteration).zfill(4)
            iteration_data_path = os.path.join(data_path, iteration_name)

            try:

                data_frame = load_processed_data_set(
                    iteration_data_path, property_type, substance_type
                )

            except FileNotFoundError:
                continue

            if len(data_frame) == 0:
                return []

            data_frames = {environment: data_frame}

            if partition_by_composition:
                data_frames = _partition_by_composition(data_frame, environment)

            for environment_label, data_frame in data_frames.items():

                data_frame = data_frame.copy()
                data_frame.reset_index(drop=True, inplace=True)

                gradient_data = defaultdict(dict)

                for header in data_frame:

                    if "d Term / d " not in header:
                        continue

                    smirks = header.split(" ")[-2]
                    attribute = header.split(" ")[-1]

                    value = data_frame[header].sum(min_count=1)

                    gradient_data[smirks][attribute] = value

                for smirks in gradient_data:

                    data_row = {
                        "Study": study_name,
                        "Property": property_title,
                        "Environment": environment_label,
                        "Iteration": iteration,
                        "Smirks": smirks,
                    }

                    if any(numpy.isnan(x) for x in gradient_data[smirks].values()):
                        continue

                    for attribute, value in gradient_data[smirks].items():
                        data_row[attribute] = value

                    data_rows.append(data_row)

    data_frame = pandas.DataFrame(data_rows)

    return data_frame
