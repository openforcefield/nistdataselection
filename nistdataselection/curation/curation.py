"""
Tools for selecting sets of molecules and corresponding measured physical properties
for use in optimizing and benchmarking molecular force fields.
"""
import functools
import logging
import math
import os
import sys
from collections import defaultdict

from openforcefield.topology import Molecule
from openforcefield.utils import UndefinedStereochemistryError
from propertyestimator import unit
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.properties import DielectricConstant, EnthalpyOfVaporization

from nistdataselection.utils import PandasDataSet
from nistdataselection.utils.utils import find_smirks_parameters, invert_dict_of_list


def _find_common_smiles_patterns(*data_sets):
    """Find the set of smiles patterns which are common to multiple
    property data sets.

    Parameters
    ----------
    data_sets: *PhysicalPropertyDataSet
        The data sets to find the common smiles patterns between.

    Returns
    -------
    set of tuple of str
        A list smiles tuples (containing the smiles in a given substance)
        which are common to all specified data sets.
    """

    assert len(data_sets) > 0

    data_set_smiles_tuples = []

    for index, data_set in enumerate(data_sets):

        smiles_tuples = set()

        # Find all unique smiles in the data set.
        for substance_id in data_set.properties:

            if len(data_set.properties[substance_id]) == 0:
                continue

            substance = data_set.properties[substance_id][0].substance
            substance_smiles = [component.smiles for component in substance.components]

            smiles_tuple = tuple(sorted(substance_smiles))
            smiles_tuples.add(smiles_tuple)

        data_set_smiles_tuples.append(smiles_tuples)

    # Find all of the smiles which are common to the requested
    # data sets.
    common_smiles_tuples = None

    for smiles_tuple_set in data_set_smiles_tuples:

        if common_smiles_tuples is None:

            common_smiles_tuples = smiles_tuple_set
            continue

        common_smiles_tuples = common_smiles_tuples.intersection(smiles_tuple_set)

    return common_smiles_tuples


def _load_data_set(directory, property_type, substance_type):
    """Loads a data set of measured physical properties of a specific
    type.

    Parameters
    ----------
    directory: str
        The path which contains the data csv files generated
        by the `process_raw_data` method.
    property_type: type of PhysicalProperty
        The property of interest.
    substance_type: SubstanceType
        The substance type of interest.

    Returns
    -------
    PhysicalPropertyDataSet
        The loaded data set.
    """

    assert os.path.isdir(directory)

    # Try to load in the pandas data file.
    file_name = f"{property_type.__name__}_{str(substance_type.value)}.csv"
    file_path = os.path.join(directory, file_name)

    if not os.path.isfile(file_path):

        raise ValueError(f"No data file could be found for " f"{substance_type} {property_type}s at {file_path}")

    data_set = PandasDataSet.from_pandas_csv(file_path, property_type)

    return data_set


def _remove_duplicates(data_set):
    """Removes duplicate properties (i.e. those measured at the same
    states) from a data set. For now, the measurement with the largest
    uncertainty is retained.

    Notes
    -----
    Temperatures are compared in kelvin to two decimal places, while
    pressures are compared in kilopascals to three decimal places.

    Warnings
    --------
    This method is not guaranteed to be deterministic.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to remove duplicates from.

    Returns
    -------
    PhysicalPropertyDataSet
        The processed data set.
    """

    properties_by_substance = {}

    for substance_id in data_set.properties:

        properties_by_substance[substance_id] = {}

        for physical_property in data_set.properties[substance_id]:

            property_type = physical_property.__class__.__name__

            # Partition the properties by type.
            if property_type not in properties_by_substance[substance_id]:
                properties_by_substance[substance_id][property_type] = {}

            # Partition the properties by state.
            temperature = physical_property.thermodynamic_state.temperature.to(unit.kelvin).magnitude

            if physical_property.thermodynamic_state.pressure is None:

                state_tuple = (f"{temperature:.2f}", f"None")

            else:

                pressure = physical_property.thermodynamic_state.pressure.to(unit.kilopascal).magnitude
                state_tuple = (f"{temperature:.2f}", f"{pressure:.3f}")

            if state_tuple not in properties_by_substance[substance_id][property_type]:

                # Handle the easy case where this is the first time a
                # property at this state has been observed.
                properties_by_substance[substance_id][property_type][state_tuple] = physical_property
                continue

            existing_property = properties_by_substance[substance_id][property_type][state_tuple]

            existing_uncertainty = math.inf if existing_property.uncertainty is None else existing_property.uncertainty
            current_uncertainty = math.inf if physical_property.uncertainty is None else physical_property.uncertainty

            base_unit = None

            if isinstance(existing_uncertainty, unit.Quantity):
                base_unit = existing_uncertainty.units

            elif isinstance(current_uncertainty, unit.Quantity):
                base_unit = current_uncertainty.units

            if base_unit is not None and isinstance(existing_uncertainty, unit.Quantity):
                existing_uncertainty = existing_uncertainty.to(base_unit).magnitude

            if base_unit is not None and isinstance(current_uncertainty, unit.Quantity):
                current_uncertainty = current_uncertainty.to(base_unit).magnitude

            if (
                math.isinf(existing_uncertainty)
                and math.isinf(current_uncertainty)
                or existing_uncertainty < current_uncertainty
            ):

                # If neither property has an uncertainty, or the existing one has
                # a lower uncertainty keep that one.
                continue

            properties_by_substance[substance_id][property_type][state_tuple] = physical_property

    # Rebuild the data set with only unique properties.
    unique_data_set = PhysicalPropertyDataSet()

    for substance_id in properties_by_substance:

        if substance_id not in unique_data_set.properties:
            unique_data_set.properties[substance_id] = []

        for property_type in properties_by_substance[substance_id]:

            for state_tuple in properties_by_substance[substance_id][property_type]:

                unique_data_set.properties[substance_id].append(
                    properties_by_substance[substance_id][property_type][state_tuple]
                )

    logging.info(
        f"{data_set.number_of_properties - unique_data_set.number_of_properties} "
        f"duplicate properties were removed."
    )

    return unique_data_set


def _apply_global_filters(data_set, temperature_range, pressure_range, allowed_elements, smiles_to_exclude):
    """Applies a set of global curation filters to the data set

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter.
    temperature_range: tuple of unit.Quantity and unit.Quantity
        The minimum and maximum temperatures.
    pressure_range: tuple of unit.Quantity and unit.Quantity
        The minimum and maximum pressures.
    allowed_elements: list of str
        A list of the allowed chemical elements.
    smiles_to_exclude: list of str
        A list of the smiles patterns to exclude.
    """

    current_number_of_properties = data_set.number_of_properties

    data_set.filter_by_temperature(min_temperature=temperature_range[0], max_temperature=temperature_range[1])
    logging.info(
        f"{current_number_of_properties - data_set.number_of_properties} "
        f"properties were outside of the temperature range and were removed."
    )

    current_number_of_properties = data_set.number_of_properties

    data_set.filter_by_pressure(min_pressure=pressure_range[0], max_pressure=pressure_range[1])
    logging.info(
        f"{current_number_of_properties - data_set.number_of_properties} "
        f"properties were outside of the pressure range and were removed."
    )

    current_number_of_properties = data_set.number_of_properties

    data_set.filter_by_elements(*allowed_elements)
    logging.info(
        f"{current_number_of_properties - data_set.number_of_properties} "
        f"properties contained unwanted elements and were removed."
    )

    # Make sure to only include molecules which have well defined stereochemsitry
    current_number_of_properties = data_set.number_of_properties

    _filter_undefined_stereochemistry(data_set)
    logging.info(
        f"{current_number_of_properties - data_set.number_of_properties} "
        f"properties contained molecules with undefined stereochemistry."
    )

    # Make sure to only include molecules which don't have a net charge.
    current_number_of_properties = data_set.number_of_properties

    _filter_non_zero_charge(data_set)
    logging.info(
        f"{current_number_of_properties - data_set.number_of_properties} " f"properties contained charged molecules."
    )

    # Exclude any excluded smiles patterns.
    current_number_of_properties = data_set.number_of_properties

    _filter_excluded_smiles(data_set, smiles_to_exclude)
    logging.info(
        f"{current_number_of_properties - data_set.number_of_properties} "
        f"properties were measured for excluded smiles patterns."
    )

    logging.info(f"The filtered data set contains {data_set.number_of_properties} properties.")


def _filter_excluded_smiles(data_set, smiles_to_exclude):
    """Filters out data points measured for excluded smiles.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    smiles_to_exclude: list of str
        The smiles patterns to exclude.
    """

    def filter_function(physical_property):

        for component in physical_property.substance.components:

            if component.smiles in smiles_to_exclude:
                return False

        return True

    data_set.filter_by_function(filter_function)


def _filter_dielectric_constants(data_set, minimum_value):
    """Filter out measured dielectric constants whose value is
    below a given threshold.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    minimum_value: simtk.unit.Quantity
        The minimum acceptable value of the
        dielectric constant.
    """

    def filter_function(physical_property):

        if not isinstance(physical_property, DielectricConstant):
            return True

        return physical_property.value >= minimum_value

    data_set.filter_by_function(filter_function)


def _filter_ionic_liquids(data_set):
    """Filters out ionic liquids.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    """

    def filter_function(physical_property):

        for component in physical_property.substance.components:

            if "." in component.smiles:
                return False

        return True

    data_set.filter_by_function(filter_function)


def _filter_non_zero_charge(data_set):
    """Filters out any molecules which have a net charge.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    """

    def filter_function(physical_property):

        for component in physical_property.substance.components:

            try:
                molecule = Molecule.from_smiles(component.smiles)
            except UndefinedStereochemistryError:
                return False

            if sum([atom.formal_charge for atom in molecule.atoms]) == 0:
                continue

            return False

        return True

    data_set.filter_by_function(filter_function)


def _filter_undefined_stereochemistry(data_set):
    """Filters out any molecules which have undefined sterochemistry.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    """

    def filter_function(physical_property):

        for component in physical_property.substance.components:

            try:
                Molecule.from_smiles(component.smiles)
            except UndefinedStereochemistryError:
                return False

        return True

    data_set.filter_by_function(filter_function)


def _filter_by_smiles_tuple(data_set, *smiles_tuples):
    """Filters out properties measured for substances which don't
     appear in the smiles tuples.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    smiles_tuples: tuple of str
        The smiles tuples to filter against.
    """

    def filter_function(physical_property):
        smiles_tuple = tuple(sorted([component.smiles for component in physical_property.substance.components]))
        return smiles_tuple in smiles_tuples

    data_set.filter_by_function(filter_function)


def _filter_by_required_smiles(data_set, required_smiles):
    """Filters out properties measured for substances which don't
     contain at least one of the required components as defined by
     it's smiles representation.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    required_smiles: list of str
        The list of smiles patterns of which at least one must appear
        in each substance.
    """

    def filter_function(physical_property):

        for component in physical_property.substance.components:

            if component.smiles not in required_smiles:
                continue

            return True

        return False

    data_set.filter_by_function(filter_function)


def _molecule_ranking_function(substance_tuple):
    """A function used to sort a list of smiles tuples in order
    of preference to include them in the final data set. Currently
    the smallest molecules which exercise the most unexercised vdW
    parameters are prioritised.

    Parameters
    ----------
    substance_tuple: tuple of tuple of str and list of str
        A tuple containing both a tuple of all of the smiles patterns
        represented by a substance, as well as the vdW smirks patterns
        which they exercise which haven't yet been fully exercised.

    Returns
    -------
    int
        The number of vdW smirks patterns exercised.
    float
        The inverse total number of atoms in the substance.
    """
    smiles_tuple, vdw_smirks = substance_tuple

    number_of_vdw_smirks = len(vdw_smirks)

    # Determine the number of vdW smirks represented by this smiles pattern.
    # We prefer smaller molecules as they will be quicker to simulate
    # and their properties should converge faster (compared to larger,
    # more flexible molecule with more degrees of freedom to sample).
    molecules = [Molecule.from_smiles(smiles) for smiles in smiles_tuple]
    number_of_atoms = sum([molecule.n_atoms for molecule in molecules])

    # Return the tuple to sort by, prioritising the number of
    # exercised vdW smirks, and then the inverse number of atoms. The
    # inverse number of atoms is used as the dictionary is being
    # reverse sorted.
    return number_of_vdw_smirks, 1.0 / number_of_atoms


def _choose_molecule_set(
    data_sets,
    properties_of_interest,
    property_order,
    vdw_smirks_to_exercise,
    desired_substances_per_property,
    desired_properties_per_smirks=2,
):
    """Selects the minimum set of molecules which (if possible) simultaneously have
    data for the first set of properties in the `property_order` list, and which exercise
    the largest number of vdW parameters.

    If such a set doesn't cover all vdW parameters, we continue through the properties
    in the order specified by `property_order` either until all vdW parameters have been
    exercised a satisfactory number of times, or all data has been considered.

    Parameters
    ----------
    data_sets: dict of type and PhysicalPropertyDataSet
        A dictionary of the data sets containing the properties of
        interest, with keys of the type of property within each data
        set.

    properties_of_interest: list of tuple of type and SubstanceType
        A list of all of the properties which are of interest to optimise against.

    property_order: list of list of tuple of type and SubstanceType
        A list of lists of property types and substance types in the order
        in which to prioritize them.

    vdw_smirks_to_exercise: list of str
        A list of those vdW smirks patterns to aim to exercise.

    desired_substances_per_property: dict of tuple and int
        The desired number of unique substances which should have data points
        for each of the properties of interest. This may not be attainable if
        a property only has limited data.

    desired_properties_per_smirks: int
        The number of each type of property which should
        ideally exercise each smirks pattern.

    Returns
    -------
    dict of str and set of str
        The smiles representations of the chosen molecules, as well as
        a set of the vdW smirks patterns which they exercise.
    """

    chosen_smiles_tuples = defaultdict(set)

    smirks_exercised_per_property = {}
    smiles_tuples_per_property = defaultdict(set)

    for property_type, _ in properties_of_interest:
        smirks_exercised_per_property[property_type] = defaultdict(int)

    # Keep track of which smiles exercise which vdW smirks.
    vdw_smirks_per_smiles_tuple = defaultdict(set)

    for property_list in property_order:

        # Find those compounds for which there is data for all of the properties of
        # interest.
        common_smiles_tuples = _find_common_smiles_patterns(
            *[data_sets[property_tuple] for property_tuple in property_list]
        )

        # Extract out all unique smiles patterns from the common tuples.
        all_smiles_patterns = set([smiles for smiles_tuple in common_smiles_tuples for smiles in smiles_tuple])

        # Find the smiles patterns which exercise each vdW smirks.
        smiles_per_vdw_smirks = find_smirks_parameters("vdW", *all_smiles_patterns)

        # Filter out the smirks patterns which are not of interest
        smirks_to_exclude = [smirks for smirks in smiles_per_vdw_smirks if smirks not in vdw_smirks_to_exercise]

        for vdw_smirks in smirks_to_exclude:
            smiles_per_vdw_smirks.pop(vdw_smirks)

        # Invert the dictionary to find those smirks exercised by each smiles pattern.
        vdw_smirks_per_smiles = invert_dict_of_list(smiles_per_vdw_smirks)

        # Use the inverted map to find which smirks are exercised per smiles tuple.
        for smiles_tuple in common_smiles_tuples:
            for smiles in smiles_tuple:
                vdw_smirks_per_smiles_tuple[smiles_tuple].update(vdw_smirks_per_smiles[smiles])

        # Invert this relationship once more.
        smiles_tuple_per_vdw_smirks = invert_dict_of_list(vdw_smirks_per_smiles_tuple)

        unexercised_smirks_per_smiles_tuples = defaultdict(set)

        # Construct a dictionary of those vdW smirks patterns which
        # haven't yet been fully exercised by the `chosen_smiles` set.
        for smirks, smiles_tuples_set in smiles_tuple_per_vdw_smirks.items():

            # The smiles set may be None in cases where no molecules
            # exercised a particular smirks pattern.
            if smiles_tuples_set is None or len(smiles_tuples_set) == 0:
                continue

            # Don't consider vdW parameters which have already been exercised
            # by the currently chosen smiles set for each of the properties of
            # interest.
            all_smirks_exercised = True

            for property_type, _ in property_list:

                if smirks_exercised_per_property[property_type][smirks] >= desired_properties_per_smirks:
                    continue

                all_smirks_exercised = False
                break

            if all_smirks_exercised is True:
                continue

            for smiles_tuple in smiles_tuples_set:

                # We don't need to consider molecules we have already chosen.
                if smiles_tuple in chosen_smiles_tuples:
                    continue

                unexercised_smirks_per_smiles_tuples[smiles_tuple].add(smirks)

        # We sort the dictionary so that molecules which will exercise the most
        # vdW parameters appear first. For molecules which exercise the same number
        # of parameters, we rank smaller molecules higher than larger ones.
        sorted_smiles_tuples = []

        for key, value in sorted(
            unexercised_smirks_per_smiles_tuples.items(), key=_molecule_ranking_function, reverse=True
        ):

            sorted_smiles_tuples.append(key)

        while len(sorted_smiles_tuples) > 0:

            # Extract the first molecule which exercises the most smirks
            # patterns at once and add it to the chosen set.
            smiles_tuple = sorted_smiles_tuples.pop(0)
            exercised_smirks = unexercised_smirks_per_smiles_tuples.pop(smiles_tuple)

            chosen_smiles_tuples[smiles_tuple] = set(vdw_smirks_per_smiles_tuple[smiles_tuple])

            for property_type, substance_type in property_list:

                for exercised_smirks_pattern in exercised_smirks:
                    smirks_exercised_per_property[property_type][exercised_smirks_pattern] += 1

                smiles_tuples_per_property[(property_type, substance_type)].add(smiles_tuple)

            # Update the dictionary to reflect that a number of
            # smirks patterns have now been exercised.
            for remaining_smiles_tuple in unexercised_smirks_per_smiles_tuples:

                for smirks in exercised_smirks:

                    if smirks not in unexercised_smirks_per_smiles_tuples[remaining_smiles_tuple]:
                        continue

                    smirks_fully_exercised = True

                    for property_type, _ in property_list:

                        if smirks_exercised_per_property[property_type][smirks] >= desired_properties_per_smirks:
                            continue

                        smirks_fully_exercised = False
                        break

                    if not smirks_fully_exercised:
                        continue

                    unexercised_smirks_per_smiles_tuples[remaining_smiles_tuple].remove(smirks)

            # Remove empty dictionary entries
            unexercised_smirks_per_smiles_tuples = {
                smiles_tuple: smirks_set
                for smiles_tuple, smirks_set in unexercised_smirks_per_smiles_tuples.items()
                if len(smirks_set) > 0
            }

            # Re-sort the smiles list by the same criteria as above.
            resorted_smiles_tuples = []

            for key, value in sorted(
                unexercised_smirks_per_smiles_tuples.items(), key=_molecule_ranking_function, reverse=True
            ):

                resorted_smiles_tuples.append(key)

            sorted_smiles_tuples = resorted_smiles_tuples

    # Include additional molecules which may exercise any vdW smirks in the set
    # if required.
    for property_list in property_order:

        required_number_extra = max(
            [
                desired_substances_per_property[property_tuple] - len(smiles_tuples_per_property[property_tuple])
                for property_tuple in property_list
            ]
        )

        # If we have already met the target number of substances for this property we
        # do not need to add any more.
        if required_number_extra <= 0:
            continue

        remaining_smiles_tuples = _find_common_smiles_patterns(
            *[data_sets[property_tuple] for property_tuple in property_list]
        )

        for property_tuple in property_list:
            remaining_smiles_tuples -= smiles_tuples_per_property[property_tuple]

        # There is no more data left to choose from.
        if len(remaining_smiles_tuples) <= 0:
            continue

        smirks_per_remaining_smirks = {
            smiles_tuple: vdw_smirks_per_smiles_tuple[smiles_tuple] for smiles_tuple in remaining_smiles_tuples
        }

        # Re-rank the smiles list by the same criteria as above.
        ranked_smiles_tuples = []

        for key, value in sorted(smirks_per_remaining_smirks.items(), key=_molecule_ranking_function, reverse=True):
            ranked_smiles_tuples.append(key)

        for _ in range(required_number_extra):

            smiles_tuple = ranked_smiles_tuples.pop(0)
            chosen_smiles_tuples[smiles_tuple] = vdw_smirks_per_smiles_tuple[smiles_tuple]

            for property_tuple in property_list:
                smiles_tuples_per_property[property_tuple].add(smiles_tuple)

            if len(ranked_smiles_tuples) <= 0:
                break

    return chosen_smiles_tuples


def _state_distance(target_state_point, state_tuple):
    """Defines a metric for how close a measured data point
    (`state_tuple`) is to a state point of interest (`target_state_point`).

    Currently this a tuple of the form (|difference in pressure|, |difference
    in temperature|), i.e., deviations from the target pressure are first
    prioritised, followed by deviations from the target temperature.

    Parameters
    ----------
    target_state_point: tuple of simtk.unit.Quantity and simtk.unit.Quantity
        A tuple containing a pressure and temperature of interest of
        the form (pressure, temperature).

    state_tuple: tuple of float and float and tuple of float and float
        The measured state point, of the form (pressure in kPa, temperature in K,
        (mole fraction 0, ..., mole fraction N).

    Returns
    -------
    tuple of float and float
        A tuple of the form (|difference in pressure|, |difference in temperature|)
    """

    pressure, temperature, mole_fractions = state_tuple

    mole_fraction_distances = [target_state_point[2][i] - mole_fractions[i] for i in range(len(mole_fractions))]
    mole_fraction_distances_sqr = [i ** 2 for i in mole_fraction_distances]

    mole_fraction_distance = sum(mole_fraction_distances_sqr)

    distance_tuple = (
        mole_fraction_distance ** 2,
        (target_state_point[1].to(unit.kilopascal).magnitude - pressure) ** 2,
        (target_state_point[0].to(unit.kelvin).magnitude - temperature) ** 2,
    )

    return distance_tuple


def _choose_data_points(data_set, properties_of_interest, target_state_points):
    """The method attempts to find a small set of data points for each
    property which are clustered around the set of conditions specified
    in the `target_state_points` input array.

    The points will be chosen so as to try and maximise the number of
    properties measured at the same condition (e.g. ideally we would
    have a data point for each property at T=298.15 and p=1atm) as this
    will maximise the chances that we can extract all properties from a
    single simulation.

    Warnings
    --------
    Currently this method will only work with pure properties of interest.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to choose data points from.
    properties_of_interest: list of tuple of type and SubstanceType
        A list of the properties which are of interest to optimise against.
    target_state_points: list of tuple of simtk.Unit.Quantity and simtk.Unit.Quantity and tuple of float
        A list of the state points for which we would ideally have data
        points for. The tuple should be of the form
        (temperature, pressure, (mole fraction 0, ..., mole fraction N))

    Returns
    -------
    PhysicalPropertyDataSet
        A data set which contains the chosen data points.
    """

    return_data_set = PhysicalPropertyDataSet()

    properties_by_components = defaultdict(list)

    for substance_id in data_set.properties:

        if len(data_set.properties[substance_id]) == 0:
            continue

        substance = data_set.properties[substance_id][0].substance
        component_tuple = tuple(sorted([component.smiles for component in substance.components]))

        properties_by_components[component_tuple].extend(data_set.properties[substance_id])

    for component_tuple in properties_by_components:

        clustered_state_points = defaultdict(list)

        property_types_by_state = defaultdict(set)
        properties_by_state = defaultdict(list)

        # We first cluster data points around the closest target state
        # according to the `state_distance` metric.
        for physical_property in properties_by_components[component_tuple]:

            temperature = physical_property.thermodynamic_state.temperature.to(unit.kelvin).magnitude
            pressure = physical_property.thermodynamic_state.pressure.to(unit.kilopascal).magnitude

            mole_fractions = tuple(
                [
                    next(iter(physical_property.substance.get_amounts(component))).value
                    for component in physical_property.substance.components
                ]
            )

            state_tuple = (round(pressure, 3), round(temperature, 2), mole_fractions)

            closest_cluster_index = -1
            shortest_cluster_distance = sys.float_info.max

            for cluster_index, target_state_point in enumerate(target_state_points):

                distance = math.sqrt(sum(_state_distance(target_state_point, state_tuple)))

                if distance >= shortest_cluster_distance:
                    continue

                closest_cluster_index = cluster_index
                shortest_cluster_distance = distance

            if state_tuple not in clustered_state_points[closest_cluster_index]:
                clustered_state_points[closest_cluster_index].append(state_tuple)

            # Keep a track of which properties (and types of properties) we
            # have for each of the measured state points.
            property_types_by_state[state_tuple].add(type(physical_property))
            properties_by_state[state_tuple].append(physical_property)

        for cluster_index, target_state_point in enumerate(target_state_points):

            # For each cluster, we try to find the state points for which we have
            # measured the most types of properties (i.e. prioritise states
            # for which we have a density, dielectric and enthalpy measurement
            # over those for which we only have a density measurement). We
            # continue to choose state points until either we have coverage
            # of all properties at the state of interest, or we have considered
            # all possible data points.
            properties_to_cover = set(property_tuple[0] for property_tuple in properties_of_interest)

            clustered_states = clustered_state_points[cluster_index]
            clustered_states = list(
                sorted(clustered_states, key=functools.partial(_state_distance, target_state_point))
            )

            chosen_states = set()

            # Iteratively consider state points which have all data points, down
            # to state points for which we only have single property measurements.
            for target_number_of_properties in reversed(range(1, len(properties_to_cover) + 1)):

                for clustered_state in clustered_states:

                    property_types_at_state = property_types_by_state[clustered_state]

                    if len(property_types_at_state) != target_number_of_properties:
                        continue

                    if len(properties_to_cover.intersection(property_types_at_state)) == 0:
                        continue

                    chosen_states.add(clustered_state)

                    properties_to_cover = properties_to_cover.symmetric_difference(
                        properties_to_cover.intersection(property_types_at_state)
                    )

            # Add the properties which were measured at the chosen state points
            # to the returned data set.
            for chosen_state in chosen_states:

                if len(properties_by_state[chosen_state]) == 0:
                    continue

                substance_id = properties_by_state[chosen_state][0].substance.identifier

                if substance_id not in return_data_set.properties:
                    return_data_set.properties[substance_id] = []

                return_data_set.properties[substance_id].extend(properties_by_state[chosen_state])

    return return_data_set


def curate_data_set(
    property_data_directory,
    property_order,
    desired_substances_per_property,
    required_smiles_to_include=None,
    smiles_to_exclude=None,
    vdw_smirks_to_exercise=None,
    minimum_data_points_per_property_per_smirks=2,
    output_data_set_path="curated_data_set.json",
):

    """The main function which will perform the
    data curation.

    Parameters
    ----------
    property_data_directory: str
        The directory which contains the processed pandas
        date sets generated by `parserawdata`.
    property_order: list of list of tuple of type and SubstanceType
        A list of lists of property types and substance types in the order
        in which to prioritize them.
    desired_substances_per_property: dict of tuple and int
        The desired number of unique substances which should have data points
        for each of the properties of interest. This may not be attainable if
        a property only has limited data.
    required_smiles_to_include: list of str, optional
        The set of smiles which must be present in the final data set
        if such data is available. In the case of data measured for pure
        substances, only compounds which appear in this list will be
        included. In the case of data measured for substances with more
        than one component, at least one of those components must appear
        in this list.
    smiles_to_exclude: list of str, optional
        The smiles patterns to exclude from the data set. This
        is useful for excluding smiles in the training set from the
        testing set.
    vdw_smirks_to_exercise: list of str, optional
        A list of those vdW smirks patterns to aim to exercise. If none,
        all vdW smirks will be attempted to be exercised.
    minimum_data_points_per_property_per_smirks: int
        The desired minimum number of data points which, for each type of property,
        exercise each VdW smirks.
    output_data_set_path: str
        The path to save the curated data set to.

    Returns
    -------
    PhysicalPropertyDataSet
        The selected data set object.
    """

    # Define the properties which we are interested in curating data for,
    # as well as the types of data we are interested in.
    properties_of_interest = set()

    for property_tuples in property_order:
        for property_tuple in property_tuples:
            properties_of_interest.add(property_tuple)

    # Define the vdW smirks to be exercised if none were provided.
    if vdw_smirks_to_exercise is None:
        vdw_smirks_to_exercise = find_smirks_parameters("vdW")

    # Define the ranges of temperatures and pressures of interest.
    # Here we choose a range of temperatures which are biologically
    # relevant (15 C - 45 C) and pressures which are close to ambient.
    temperature_range = (288.15 * unit.kelvin, 318.15 * unit.kelvin)
    pressure_range = (0.95 * unit.atmosphere, 1.05 * unit.atmosphere)

    # Specify more exactly those state points which would be of interest
    # to fit against. These are currently tuples of temperature, pressure,
    # and a tuple of the mole fractions of each of the components.
    target_state_points = [
        (298.15 * unit.kelvin, 101.325 * unit.kilopascal, (0.25, 0.75)),
        (298.15 * unit.kelvin, 101.325 * unit.kilopascal, (0.75, 0.25)),
    ]

    # Define the elements that we are interested in. Here we only allow
    # a subset of those elements for which smirnoff99Frosst has parameters
    # for.
    allowed_elements = ["H", "N", "C", "O", "S", "F", "Cl", "Br", "I"]

    # Define a minimum dielectric constant threshold so as not to try
    # try to reproduce values which cannot be simulated with accuracy.
    minimum_dielectric_value = 10.0 * unit.dimensionless

    # Define whether or not to allow ionic liquids.
    allow_ionic_liquids = False

    # Load and apply basic filters to the data sets of interest.
    data_sets = {}

    for property_type, substance_type in properties_of_interest:

        # Load the full data sets from the processed data files, and
        # remove any duplicate properties according to `_remove_duplicates`.
        logging.info(f"Loading the {substance_type.value} {property_type.__name__} data set.")

        data_set = _load_data_set(property_data_directory, property_type, substance_type)
        data_set = _remove_duplicates(data_set)

        # Retain only properties which contain any required components.
        if required_smiles_to_include is not None:
            _filter_by_required_smiles(data_set, required_smiles_to_include)

        # Apply the a number of global filters, such as excluding data points
        # outside of the chosen temperature and pressure ranges.
        _apply_global_filters(data_set, temperature_range, pressure_range, allowed_elements, smiles_to_exclude)

        # Optionally filter by ionic liquids.
        if allow_ionic_liquids is False:
            _filter_ionic_liquids(data_set)

        # Additionally filter out any measured dielectric constants which are too low
        # and may be difficult to simulate.
        if property_type == DielectricConstant:

            current_number_of_properties = data_set.number_of_properties

            _filter_dielectric_constants(data_set, minimum_dielectric_value)

            logging.info(
                f"{current_number_of_properties - data_set.number_of_properties} "
                f"dielectric properties had values less than {minimum_dielectric_value} "
                f"and were removed."
            )

        data_sets[(property_type, substance_type)] = data_set

        logging.info(f"Finished loading the {substance_type} {property_type.__name__} data set.")

    # Due to ambiguity about at which pressure the enthalpy of vaporisation
    # was collected (no pressure is recorded in any of the ThermoML archives,
    # a decision was made to assume a pressure of 1 atm.
    for property_type, substance_type in data_sets:

        if not property_type == EnthalpyOfVaporization:
            continue

        for substance_id in data_sets[(property_type, substance_type)].properties:
            for physical_property in data_sets[(property_type, substance_type)].properties[substance_id]:
                physical_property.thermodynamic_state.pressure = 1.0 * unit.atmosphere

    # Choose a set of molecules which give a good coverage of
    # the vdW parameters which will be optimised against the
    # properties of interest.
    chosen_smiles_tuples = _choose_molecule_set(
        data_sets,
        properties_of_interest,
        property_order,
        vdw_smirks_to_exercise,
        desired_substances_per_property,
        minimum_data_points_per_property_per_smirks,
    )

    logging.info(f'Chosen smiles tuples: {" ".join(map(str, chosen_smiles_tuples.keys()))}')

    # Merge the multiple property data sets into a single object
    final_data_set = PhysicalPropertyDataSet()

    if len(chosen_smiles_tuples) > 0:

        for data_set in data_sets.values():
            final_data_set.merge(data_set)

        _filter_by_smiles_tuple(final_data_set, *chosen_smiles_tuples)

    # Finally, choose only a minimal set of data points from the full
    # filtered set which are concentrated on the state points specified
    # by the `target_state_points` array.
    final_data_set = _choose_data_points(final_data_set, properties_of_interest, target_state_points)

    # Save the final data set in a form consumable by the propertyestimator.
    with open(output_data_set_path, "w") as file:
        file.write(final_data_set.json())

    return final_data_set
