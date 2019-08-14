"""
Records the tools and decisions used to select NIST data for curation.
"""
import functools
import logging
import math
import os
import sys
from collections import defaultdict

import numpy
from openforcefield.topology import Molecule
from propertyestimator import unit
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.properties import Density, DielectricConstant, EnthalpyOfVaporization
from propertyestimator.utils import setup_timestamp_logging

from nistdataselection.utils import PandasDataSet
from nistdataselection.utils.utils import SubstanceType, find_smirks_parameters, cached_smirks_parameters


def _find_common_smiles_patterns(*data_sets):
    """Find the set of smiles patterns which are common to multiple
    property data sets.

    Parameters
    ----------
    data_sets: *PhysicalPropertyDataSet
        The data sets to find the common smiles patterns between.

    Returns
    -------
    set of str
        The smiles patterns which are common to all specified
        data sets.
    """

    assert len(data_sets) > 0

    data_set_smiles = []

    for index, data_set in enumerate(data_sets):

        smiles = set()

        # Find all unique smiles in the data set.
        for substance_id in data_set.properties:

            if len(data_set.properties[substance_id]) == 0:
                continue

            substance = data_set.properties[substance_id][0].substance

            for component in substance.components:
                smiles.add(component.smiles)

        data_set_smiles.append(smiles)

    # Find all of the smiles which are common to the requested
    # data sets.
    common_smiles = None

    for smiles_set in data_set_smiles:

        if common_smiles is None:

            common_smiles = smiles_set
            continue

        common_smiles = common_smiles.intersection(smiles_set)

    return common_smiles


def _load_data_set(directory, property_type, substance_type):
    """Loads a data set of measured physical properties of a specific
    type.

    Parameters
    ----------
    directory: str
        The path which contains the data csv files generated
        by the `parse_raw_data` method.
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
    file_name = f'{property_type.__name__}_{str(substance_type.value)}.csv'
    file_path = os.path.join(directory, file_name)

    if not os.path.isfile(file_path):

        raise ValueError(f'No data file could be found for '
                         f'{substance_type} {property_type}s at {file_path}')

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

                state_tuple = (f'{temperature:.2f}', f'None')

            else:

                pressure = physical_property.thermodynamic_state.pressure.to(unit.kilopascal).magnitude
                state_tuple = (f'{temperature:.2f}', f'{pressure:.3f}')

            if state_tuple not in properties_by_substance[substance_id][property_type]:

                # Handle the easy case where this is the first time a
                # property at this state has been observed.
                properties_by_substance[substance_id][property_type][state_tuple] = physical_property
                continue

            existing_property = properties_by_substance[substance_id][property_type][state_tuple]

            existing_uncertainty = (math.inf if existing_property.uncertainty is None else
                                    existing_property.uncertainty)

            current_uncertainty = (math.inf if physical_property.uncertainty is None else
                                   physical_property.uncertainty)

            base_unit = None

            if isinstance(existing_uncertainty, unit.Quantity):
                base_unit = existing_uncertainty.units

            elif isinstance(current_uncertainty, unit.Quantity):
                base_unit = current_uncertainty.units

            if base_unit is not None and isinstance(existing_uncertainty, unit.Quantity):
                existing_uncertainty = existing_uncertainty.to(base_unit).magnitude

            if base_unit is not None and isinstance(current_uncertainty, unit.Quantity):
                current_uncertainty = current_uncertainty.to(base_unit).magnitude

            if (math.isinf(existing_uncertainty) and math.isinf(current_uncertainty) or
                existing_uncertainty < current_uncertainty):

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
                    properties_by_substance[substance_id][property_type][state_tuple])

    logging.info(f'{data_set.number_of_properties - unique_data_set.number_of_properties} '
                 f'duplicate properties were removed.')

    return unique_data_set


def _apply_global_filters(data_set, temperature_range, pressure_range, allowed_elements):
    """

    Parameters
    ----------
    data_set
    temperature_range
    pressure_range
    allowed_elements

    Returns
    -------

    """

    current_number_of_properties = data_set.number_of_properties

    data_set.filter_by_temperature(min_temperature=temperature_range[0],
                                   max_temperature=temperature_range[1])
    logging.info(f'{current_number_of_properties - data_set.number_of_properties} '
                 f'properties were outside of the temperature range and were removed.')

    current_number_of_properties = data_set.number_of_properties

    data_set.filter_by_pressure(min_pressure=pressure_range[0],
                                max_pressure=pressure_range[1])
    logging.info(f'{current_number_of_properties - data_set.number_of_properties} '
                 f'properties were outside of the pressure range and were removed.')

    current_number_of_properties = data_set.number_of_properties

    data_set.filter_by_elements(*allowed_elements)
    logging.info(f'{current_number_of_properties - data_set.number_of_properties} '
                 f'properties contained unwanted elements and were removed.')

    logging.info(f'The filtered data set contains {data_set.number_of_properties} properties.')


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

            if '.' in component.smiles:
                return False

        return True

    data_set.filter_by_function(filter_function)


def _extract_min_max_median_temperature_set(data_set):
    """For a given data set, this method filters out all but the
    data measured at the minimum, median, and maximum temperatures.

    Notes
    -----
    This method should *only* be used on data sets which
    contain a single type of property for now.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter.

    Returns
    -------
    PhysicalPropertyDataSet
        The filtered data set.
    """

    filtered_set = PandasDataSet()

    for substance_id in data_set.properties:

        temperatures = []

        for physical_property in data_set.properties[substance_id]:
            temperatures.append(physical_property.thermodynamic_state.temperature.to(unit.kelvin).magnitude)

        if len(temperatures) <= 0:
            continue

        temperatures = numpy.array(temperatures)

        filtered_set.properties[substance_id] = []

        minimum_temperature_index = numpy.argmin(temperatures)
        median_temperature_index = numpy.argsort(temperatures)[len(temperatures) // 2]
        maximum_temperature_index = numpy.argmax(temperatures)

        temperature_set = set()

        temperature_set.add(minimum_temperature_index)
        temperature_set.add(median_temperature_index)
        temperature_set.add(maximum_temperature_index)

        for index in set(temperature_set):
            filtered_set.properties[substance_id].append(data_set.properties[substance_id][index])

    return filtered_set


def _choose_molecule_set(data_sets, properties_of_interest, desired_properties_per_smirks=2):
    """Selects the minimum set of molecules which (if possible) simultaneously have
    data for all three properties, and exercise the largest number of vdW
    parameters.

    If such a set doesn't cover all vdW parameters, we then expand the set with
    molecules which partially meet the above criteria, but only have data for
    enthalpies of vaporization and densities, or densities and dielectrics.

    Finally, relax the criteria to include molecules which only have data for enthalpy
    of vaporisation or density data.

    Parameters
    ----------
    data_sets: dict of type and PhysicalPropertyDataSet
        A dictionary of the data sets containing the properties of
        interest, with keys of the type of property within each data
        set.

    properties_of_interest: list of tuple of type and SubstanceType
        A list of the properties which are of interest to optimise against.

    desired_properties_per_smirks: int
        The number of each type of property which should
        ideally exercise each smirks pattern.

    Returns
    -------
    dict of str and set of str
        The smiles representations of the chosen molecules, as well as
        a set of the vdW smirks patterns which they exercise.
    """

    chosen_smiles = defaultdict(set)
    smirks_exercised_per_property = {}

    for property_type, _ in properties_of_interest:
        smirks_exercised_per_property[property_type] = defaultdict(int)

    # Define the order of preference for which data molecules should have,
    # as explained above.
    property_order = [
        [
            # Ideally we want molecules for which we have data for
            # all three properties of interest.
            (Density, SubstanceType.Pure),
            (DielectricConstant, SubstanceType.Pure),
            (EnthalpyOfVaporization, SubstanceType.Pure)
        ],
        [
            # If that isn't possible, we'd like molecules for which we
            # have at least densities and enthalpies of vaporization...
            (Density, SubstanceType.Pure),
            (EnthalpyOfVaporization, SubstanceType.Pure)
        ],
        # [
        #     # and molecules for which we have at least densities and
        #     # dielectric constant
        #     (Density, SubstanceType.Pure),
        #     (DielectricConstant, SubstanceType.Pure),
        # ],
        [
            # Finally, fall back to molecules for which the is only
            # data for the enthalpy of vaporisation...
            (EnthalpyOfVaporization, SubstanceType.Pure)
        ],
        # [
        #     # or the density.
        #     (Density, SubstanceType.Pure),
        # ]
    ]

    for property_list in property_order:

        # Find those compounds for which there is data for all of the properties of
        # interest.
        common_smiles = _find_common_smiles_patterns(
            *[data_sets[property_tuple] for property_tuple in property_list]
        )

        smiles_per_vdw_smirks = find_smirks_parameters('vdW', *common_smiles)

        unexercised_smirks_per_smiles = defaultdict(set)

        # Construct a dictionary of those vdW smirks patterns which
        # haven't yet been exercised by the `chosen_smiles` set.
        for smirks, smiles_set in smiles_per_vdw_smirks.items():

            # The smiles set may be None in cases where no molecules
            # exercised a particular smirks pattern.
            if smiles_set is None:
                continue

            # Don't consider vdW parameters which have already been exercised
            # by the currently chosen smiles set for each of the properties of
            # interest.
            all_properties_exercised = True

            for property_type, _ in property_list:

                if smirks_exercised_per_property[property_type][smirks] >= desired_properties_per_smirks:
                    continue

                all_properties_exercised = False
                break

            if all_properties_exercised is True:
                continue

            for smiles in smiles_set:

                # We don't need to consider molecules we have already chosen.
                if smiles in chosen_smiles:
                    continue

                unexercised_smirks_per_smiles[smiles].add(smirks)

        # We sort the dictionary so that molecules which will exercise the most
        # vdW parameters appear first. For molecules which exercise the same number
        # of parameters, we rank smaller molecules higher than larger ones.
        sorted_smiles = []
        printed_list = []

        def sorting_function(key_value_pair):

            smiles, vdw_smirks = key_value_pair

            number_of_vdw_smirks = len(vdw_smirks)

            # Determine the number of represented by this smiles pattern.
            # We prefer smaller molecules as they will be quicker to simulate
            # and their properties should converge faster (compared to larger,
            # more flexible molecule with more degrees of freedom to sample).
            molecule = Molecule.from_smiles(smiles)
            number_of_atoms = molecule.n_atoms

            # Return the tuple to sort by, prioritising the number of
            # exercised smirks, and then the inverse number of atoms. The
            # inverse number of atoms is used as the dictionary is being
            # reverse sorted.
            return number_of_vdw_smirks, 1.0 / number_of_atoms

        for key, value in sorted(unexercised_smirks_per_smiles.items(),
                                 key=sorting_function, reverse=True):
            sorted_smiles.append(key)

            molecule = Molecule.from_smiles(key)
            printed_list.append((key, len(value), 1.0 / molecule.n_atoms))

        while len(sorted_smiles) > 0:

            # Extract the first molecule which exercises the most smirks
            # patterns at once and add it to the chosen set.
            smiles = sorted_smiles.pop(0)
            exercised_smirks = unexercised_smirks_per_smiles.pop(smiles)

            chosen_smiles[smiles] = set([smirks for smirks, values in
                                         find_smirks_parameters('vdW', smiles).items() if len(values) > 0])

            for exercised_smirks_pattern in exercised_smirks:
                for property_type, _ in property_list:
                    smirks_exercised_per_property[property_type][exercised_smirks_pattern] += 1

            # Update the dictionary to reflect that a number of
            # smirks patterns have now been exercised.
            for remaining_smiles in unexercised_smirks_per_smiles:

                for smirks in exercised_smirks:

                    if smirks not in unexercised_smirks_per_smiles[remaining_smiles]:
                        continue

                    smirks_fully_exercised = True

                    for property_type, _ in property_list:

                        if smirks_exercised_per_property[property_type][smirks] >= desired_properties_per_smirks:
                            continue

                        smirks_fully_exercised = False
                        break

                    if not smirks_fully_exercised:
                        continue

                    unexercised_smirks_per_smiles[remaining_smiles].remove(smirks)

            # Remove empty dictionary entries
            unexercised_smirks_per_smiles = {smiles: smirks_set for smiles, smirks_set in
                                             unexercised_smirks_per_smiles.items() if len(smirks_set) > 0}

            # Re-sort the smiles list by the same criteria as above.
            resorted_smiles = []

            for key, value in sorted(unexercised_smirks_per_smiles.items(), key=sorting_function, reverse=True):
                resorted_smiles.append(key)

            sorted_smiles = resorted_smiles

    return chosen_smiles


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
    target_state_points: list of tuple of simtk.Unit.Quantity and simtk.Unit.Quantity
        A list of the state points for which we would ideally have data
        points for. The tuple should be of the form (temperature, pressure)

    Returns
    -------
    PhysicalPropertyDataSet
        A data set which contains the chosen data points.
    """

    return_data_set = PhysicalPropertyDataSet()

    def state_distance(target_state_point, state_tuple):
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

        state_tuple: tuple of float and float
            The measured state point, of the form (pressure in kPa, temperature in K).

        Returns
        -------
        tuple of float and float
            A tuple of the form (|difference in pressure|, |difference in temperature|)
        """

        pressure, temperature = state_tuple

        distance_tuple = ((target_state_point[1].to(unit.kilopascal).magnitude - pressure) ** 2,
                          (target_state_point[0].to(unit.kelvin).magnitude - temperature) ** 2)

        return distance_tuple

    for substance_id in data_set.properties:

        return_data_set.properties[substance_id] = []

        clustered_state_points = defaultdict(list)

        property_types_by_state = defaultdict(set)
        properties_by_state = defaultdict(list)

        # We first cluster data points around the closest target state
        # according to the `state_distance` metric.
        for physical_property in data_set.properties[substance_id]:

            temperature = physical_property.thermodynamic_state.temperature.to(unit.kelvin).magnitude
            pressure = physical_property.thermodynamic_state.pressure.to(unit.kilopascal).magnitude

            state_tuple = (round(pressure, 3), round(temperature, 2))

            closest_cluster_index = -1
            shortest_cluster_distance = sys.float_info.max

            for cluster_index, target_state_point in enumerate(target_state_points):

                distance = math.sqrt((target_state_point[0].to(unit.kelvin).magnitude - temperature) ** 2 +
                                     (target_state_point[1].to(unit.kilopascal).magnitude - pressure) ** 2)

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
            clustered_states = list(sorted(clustered_states, key=functools.partial(state_distance, target_state_point)))

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
                        properties_to_cover.intersection(property_types_at_state))

            # Add the properties which were measured at the chosen state points
            # to the returned data set.
            for chosen_state in chosen_states:
                return_data_set.properties[substance_id].extend(properties_by_state[chosen_state])

    return return_data_set


def curate_data_set(property_data_directory, output_data_set_path='curated_data_set.json'):
    """The main function which will perform the
    data curation.

    Parameters
    ----------
    property_data_directory: str
        The directory which contains the processed pandas
        date sets generated by `parserawdata`.
    output_data_set_path: str
        The path to save the curated data set to.
    report_type: ReportType
        The type of report to create.
    report_path: str
        The path pointing to where to store the report.
    """

    setup_timestamp_logging()

    # Define the properties which we are interested in curating data for,
    # as well as the types of data we are interested in.
    properties_of_interest = [
        (Density, SubstanceType.Pure),
        (DielectricConstant, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure)
    ]

    # Define the ranges of temperatures and pressures of interest.
    # Here we choose a range of temperatures which are biologically
    # relevant (15 C - 45 C) and pressures which are close to ambient.
    temperature_range = (288.15 * unit.kelvin, 318.15 * unit.kelvin)
    pressure_range = (0.95 * unit.atmosphere, 1.05 * unit.atmosphere)

    # Specify more exactly those state points which would be of interest
    # to fit against
    target_state_points = [
        (298.15 * unit.kelvin, 101.325 * unit.kilopascal),
        (318.15 * unit.kelvin, 101.325 * unit.kilopascal)
    ]

    # Define the elements that we are interested in. Here we only allow
    # those elements for which smirnoff99Frosst has parameters for.
    allowed_elements = ['H', 'N', 'C', 'O', 'S', 'P', 'F',
                        'Cl', 'Br', 'I', 'Na', 'K', 'Ca']

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
        logging.info(f'Loading the {substance_type.value} {property_type.__name__} data set.')

        data_set = _load_data_set(property_data_directory, property_type, substance_type)
        data_set = _remove_duplicates(data_set)

        # Apply the a number of global filters, such as excluding data points
        # outside of the chosen temperature and pressure ranges.
        _apply_global_filters(data_set,
                              temperature_range,
                              pressure_range,
                              allowed_elements)

        # Extract only data points which are either at the extremes,
        # or in the middle of the temperature range. This should yield
        # at most three data points per property per molecule.
        # data_set = _extract_min_max_median_temperature_set(data_set)

        # Optionally filter by ionic liquids.
        if allow_ionic_liquids is False:
            _filter_ionic_liquids(data_set)

        # Additionally filter out any measured dielectric constants which are too low
        # and may be difficult to simulate.
        if property_type == DielectricConstant:

            current_number_of_properties = data_set.number_of_properties

            _filter_dielectric_constants(data_set, minimum_dielectric_value)

            logging.info(f'{current_number_of_properties - data_set.number_of_properties} '
                         f'dielectric properties had values less than {minimum_dielectric_value} '
                         f'and were removed.')

        data_sets[(property_type, substance_type)] = data_set

        logging.info(f'Finished loading the {substance_type} {property_type.__name__} data set.')

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
    chosen_smiles = _choose_molecule_set(data_sets, properties_of_interest)
    logging.info(f'Chosen smiles: {" ".join(chosen_smiles.keys())}')

    # Merge the multiple property data sets into a single object
    final_data_set = PhysicalPropertyDataSet()

    for data_set in data_sets.values():
        final_data_set.merge(data_set)

    final_data_set.filter_by_smiles(*chosen_smiles)

    # Finally, choose only a minimal set of data points from the full
    # filtered set which are concentrated on the state points specified
    # by the `target_state_points` array.
    final_data_set = _choose_data_points(final_data_set, properties_of_interest, target_state_points)

    # Save the final data set in a form consumable by force balance.
    with open(output_data_set_path, 'w') as file:
        file.write(final_data_set.json())


def _main():
    """A utility function for calling this script directly, which
    expects that the data files generated by the `parserawdata`
    script are located in the '~/property_data' directory.
    """

    # home_directory = os.path.expanduser("~")
    # data_directory = os.path.join(home_directory, 'property_data')

    import json

    # Check to see if we have already cached which vdW smirks will be
    # assigned to which molecules. This can significantly speed up this
    # script on multiple runs.
    cached_smirks_file_name = 'cached_smirks_parameters.json'

    try:

        with open(cached_smirks_file_name) as file:
            cached_smirks_parameters.update(json.load(file))

    except (json.JSONDecodeError, FileNotFoundError):
        pass

    data_directory = 'data_sets'

    curate_data_set(data_directory)

    # Cache the smirks which will be assigned to the different molecules
    # to speed up future runs.
    with open(cached_smirks_file_name, 'w') as file:
        json.dump(cached_smirks_parameters, file)


if __name__ == '__main__':
    _main()
