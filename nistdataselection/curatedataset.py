"""
Records the tools and decisions used to select NIST data for curation.
"""
import logging
import math
import os
import re
from collections import defaultdict
from enum import Enum

import numpy
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines import smirnoff
from openforcefield.utils import UndefinedStereochemistryError
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.properties import Density, DielectricConstant, EnthalpyOfVaporization
from propertyestimator.utils import setup_timestamp_logging
from simtk import unit
from tabulate import tabulate

from nistdataselection.utils import PandasDataSet


class SubstanceType(Enum):
    """An enum which encodes the names used for substances
    with different numbers of components.
    """
    Pure = 'pure'
    Binary = 'binary'
    Ternary = 'ternary'


_cached_smirks_parameters = {}


def _find_smirks_parameters(parameter_tag='vdW', *smiles_patterns):
    """Finds those force field parameters with a given tag which
    would be assigned to a specified set of molecules defined by
    the their smiles patterns.

    Parameters
    ----------
    parameter_tag: str
        The tag of the force field parameters to find.
    smiles_patterns: str
        The smiles patterns to assign the force field parameters
        to.

    Returns
    -------
    dict of str and list of str
        A dictionary with keys of parameter smirks patterns, and
        values of lists of smiles patterns which would utilize
        those parameters.
    """

    force_field = smirnoff.ForceField('smirnoff99Frosst-1.0.9.offxml')
    parameter_handler = force_field.get_parameter_handler(parameter_tag)

    smiles_by_parameter_smirks = {}

    # Initialize the array with all possible smirks pattern
    # to make it easier to identify which are missing.
    for parameter in parameter_handler.parameters:

        if parameter.smirks in smiles_by_parameter_smirks:
            continue

        smiles_by_parameter_smirks[parameter.smirks] = set()

    # Populate the dictionary using the open force field toolkit.
    for smiles in smiles_patterns:

        if smiles not in _cached_smirks_parameters:

            try:
                molecule = Molecule.from_smiles(smiles)
            except UndefinedStereochemistryError:
                # Skip molecules with undefined stereochemistry.
                continue

            topology = Topology.from_molecules([molecule])
            _cached_smirks_parameters[smiles] = force_field.label_molecules(topology)[0]

        parameters_with_tag = _cached_smirks_parameters[smiles][parameter_tag]

        for parameter in parameters_with_tag.values():
            smiles_by_parameter_smirks[parameter.smirks].add(smiles)

    return smiles_by_parameter_smirks


def _count_parameters_per_molecule(parameter_tag='vdW', *smiles_patterns):
    """Returns the frequency that a certain number of parameters with a given
     tag (e.g. 'vdW') get assigned to a list of molecules defined by their
     smiles patterns.

    Parameters
    ----------
    parameter_tag: str
        The parameter tag.
    smiles_patterns: str
        The smiles patterns which define the list of molecules.

    Returns
    -------
    dict of int and int
        The counted frequencies.
    """

    smiles_by_parameter_smirks = _find_smirks_parameters(parameter_tag, *smiles_patterns)

    parameter_smirks_by_smiles = defaultdict(list)

    for smirks_pattern in smiles_by_parameter_smirks:
        for smiles_pattern in smiles_by_parameter_smirks[smirks_pattern]:
            parameter_smirks_by_smiles[smiles_pattern].append(smirks_pattern)

    counts = defaultdict(int)

    for smiles_pattern in parameter_smirks_by_smiles:

        number_of_parameters = len(parameter_smirks_by_smiles[smiles_pattern])
        counts[number_of_parameters] += 1

    return counts


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
            temperature = physical_property.thermodynamic_state.temperature.value_in_unit(unit.kelvin)

            if physical_property.thermodynamic_state.pressure is None:

                state_tuple = (f'{temperature:.2f}', f'None')

            else:

                pressure = physical_property.thermodynamic_state.pressure.value_in_unit(unit.kilopascal)
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
                base_unit = existing_uncertainty.unit

            elif isinstance(current_uncertainty, unit.Quantity):
                base_unit = current_uncertainty.unit

            if base_unit is not None and isinstance(existing_uncertainty, unit.Quantity):
                existing_uncertainty = existing_uncertainty.value_in_unit(base_unit)

            if base_unit is not None and isinstance(current_uncertainty, unit.Quantity):
                current_uncertainty = current_uncertainty.value_in_unit(base_unit)

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

            if '.' in component.smiles and ('+' in component.smiles or '-' in component.smiles):
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
            temperatures.append(physical_property.thermodynamic_state.temperature.value_in_unit(unit.kelvin))

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


def _choose_molecule_set(data_sets, properties_of_interest):
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

    Returns
    -------
    dict of str and set of str
        The smiles representations of the chosen molecules, as well as
        a set of the vdW smirks patterns which they exercise.
    """

    chosen_smiles = defaultdict(set)
    smirks_exercised_per_property = {}

    for property_type, _ in properties_of_interest:
        smirks_exercised_per_property[property_type] = set()

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
        [
            # and molecules for which we have at least densities and
            # dielectric constant
            (Density, SubstanceType.Pure),
            (DielectricConstant, SubstanceType.Pure),
        ],
        [
            # Finally, fall back to molecules for which the is only
            # data for the enthalpy of vaporisation...
            (EnthalpyOfVaporization, SubstanceType.Pure)
        ],
        [
            # or the density.
            (Density, SubstanceType.Pure),
        ]
        # TODO: Do we want to fit against molecules for which we ONLY
        #       have the dielectric constant?
    ]

    for property_list in property_order:

        # Find those compounds for which there is data for all of the properties of
        # interest.
        common_smiles = _find_common_smiles_patterns(
            *[data_sets[property_tuple] for property_tuple in property_list]
        )

        smiles_per_vdw_smirks = _find_smirks_parameters('vdW', *common_smiles)

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

                if smirks in smirks_exercised_per_property[property_type]:
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

        # Sort the dictionary keys so that molecules which will exercise the most
        # vdW parameters appear first.
        sorted_smiles = []

        for key, value in sorted(unexercised_smirks_per_smiles.items(),
                                 key=lambda x: len(x[1]), reverse=True):
            sorted_smiles.append(key)

        while len(sorted_smiles) > 0:

            # Extract the first molecule which exercises the most smirks
            # patterns at once and add it to the chosen set.
            smiles = sorted_smiles.pop(0)
            exercised_smirks = unexercised_smirks_per_smiles.pop(smiles)

            chosen_smiles[smiles] = set([smirks for smirks, values in
                                         _find_smirks_parameters('vdW', smiles).items() if len(values) > 0])

            for exercised_smirks_pattern in exercised_smirks:
                for property_type, _ in property_list:
                    smirks_exercised_per_property[property_type].add(exercised_smirks_pattern)

            # Update the dictionary to reflect that a number of
            # smirks patterns have now been exercised.
            for remaining_smiles in unexercised_smirks_per_smiles:
                unexercised_smirks_per_smiles[remaining_smiles] = unexercised_smirks_per_smiles[
                    remaining_smiles].difference(exercised_smirks)

            # Remove empty dictionary entries
            unexercised_smirks_per_smiles = {smiles: smirks_set for smiles, smirks_set in
                                             unexercised_smirks_per_smiles.items() if len(smirks_set) > 0}

            # Re-sort the smiles list.
            resorted_smiles = []

            for key, value in sorted(unexercised_smirks_per_smiles.items(), key=lambda x: len(x[1]), reverse=True):
                resorted_smiles.append(key)

            sorted_smiles = resorted_smiles

    # Construct a dictionary which expresses which data we have
    # for each smirks pattern.
    properties_exercised_by_smirks = defaultdict(set)

    for property_type, smirks_list in smirks_exercised_per_property.items():
        for smirks_pattern in smirks_list:
            properties_exercised_by_smirks[smirks_pattern].add(property_type)

    # Print information about those vdw parameters for which
    # no matched smiles patterns were found.
    all_vdw_smirks = set(_find_smirks_parameters('vdW').keys())

    for smirks in all_vdw_smirks:

        if smirks not in properties_exercised_by_smirks:
            continue

        property_string = ', '.join([property_type.__name__ for property_type in
                                     properties_exercised_by_smirks[smirks]])

        logging.info(f'{smirks} was exercised by {property_string}.')

    for smirks in all_vdw_smirks:

        if smirks in properties_exercised_by_smirks:
            continue

        logging.info(f'{smirks} was not exercised.')

    return chosen_smiles


def _print_chosen_set(chosen_smiles, data_sets, properties_of_interest):
    """Print the chosen data to the terminal.

    Parameters
    ----------
    chosen_smiles: dict of str and set of str
        The smile patterns of the chosen molecules, and the
        vdW smirks patterns they exercise.
    data_sets: dict of tuple of type and SubstanceType and PhysicalPropertyDataSet
        The data sets containing the properties of interest.
    properties_of_interest: list of tuple of type and SubstanceType
        The properties of interest.
    """
    print('\n')
    # Print information about the chosen set
    for smiles_pattern in chosen_smiles:

        print(''.join(['-'] * 120))
        print(f'\nSMILES: {smiles_pattern}\n')
        print(f'VDW SMIRKS EXERCISED: {" ".join(chosen_smiles[smiles_pattern])}')

        for property_type, substance_type in properties_of_interest:

            property_name = ' '.join(re.sub('([A-Z][a-z]+)', r' \1',
                                            re.sub('([A-Z]+)', r' \1', property_type.__name__)).split())

            print(f'\n{str(substance_type.value).upper()} {property_name.upper()} Data\n')

            data_set = data_sets[(property_type, substance_type)]

            pandas_data_frame = PandasDataSet.to_pandas_data_frame(data_set)
            pandas_data_frame = pandas_data_frame.loc[pandas_data_frame['Component 1'] == smiles_pattern]

            pandas_data_frame = pandas_data_frame[['Temperature (K)', 'Pressure (kPa)', 'Source']]

            pandas_data_frame.sort_values('Temperature (K)')
            pandas_data_frame.sort_values('Pressure (kPa)')

            print(tabulate(pandas_data_frame, headers='keys', tablefmt='psql', showindex=False))

    print(''.join(['-'] * 120))
    print('\n')


def curate_data_set(property_data_directory):
    """The main function which will perform the
    data curation.

    Parameters
    ----------
    property_data_directory: str
        The directory which contains the processed pandas
        date sets generated by `parserawdata`.
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

    filtered_data_sets_directory = 'filtered_data_sets'
    os.makedirs(filtered_data_sets_directory, exist_ok=True)

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
        # or in the middle of the temperature range.
        data_set = _extract_min_max_median_temperature_set(data_set)

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

    # Choose a set of molecules which give a good coverage of
    # the vdW parameters which will be optimised against the
    # properties of interest.
    chosen_smiles = _choose_molecule_set(data_sets, properties_of_interest)
    logging.info(f'Chosen smiles: {" ".join(chosen_smiles.keys())}')

    # Print the chosen molecule set and the corresponding data to the terminal.
    _print_chosen_set(chosen_smiles, data_sets, properties_of_interest)

    # Merge the multiple property data sets into a single object
    final_data_set = PhysicalPropertyDataSet()

    # TODO: Migrate to the PhysicalPropertyDataSet class.
    def filter_by_smiles(physical_property):
        for component in physical_property.substance.components:
            if component.smiles in chosen_smiles:
                continue
            return False
        return True

    for data_set in data_sets.values():
        data_set.filter_by_function(filter_by_smiles)
        final_data_set.merge(data_set)

    # Save the final data set in a form consumable by force balance.
    with open('curated_data_set.json', 'w') as file:
        file.write(final_data_set.json())


def _main():
    """A utility function for calling this script directly, which
    expects that the data files generated by the `parserawdata`
    script are located in the '~/property_data' directory.
    """

    # home_directory = os.path.expanduser("~")
    # data_directory = os.path.join(home_directory, 'property_data')

    data_directory = 'data_sets'

    curate_data_set(data_directory)


if __name__ == '__main__':
    _main()
