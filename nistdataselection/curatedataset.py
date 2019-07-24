"""
Records the tools and decisions used to select NIST data for curation.
"""
import logging
import math
import os
from collections import defaultdict
from enum import Enum

from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines import smirnoff
from openforcefield.utils import UndefinedStereochemistryError
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.properties import Density, DielectricConstant, EnthalpyOfVaporization
from propertyestimator.utils import setup_timestamp_logging
from simtk import unit

from nistdataselection.utils import PandasDataSet


class SubstanceType(Enum):
    """An enum which encodes the names used for substances
    with different numbers of components.
    """
    Pure = 'pure'
    Binary = 'binary'
    Ternary = 'ternary'


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

        try:
            molecule = Molecule.from_smiles(smiles)
        except UndefinedStereochemistryError:
            # Skip molecules with undefined stereochemistry.
            continue

        topology = Topology.from_molecules([molecule])

        assigned_parameters = force_field.label_molecules(topology)[0]
        parameters_with_tag = assigned_parameters[parameter_tag]

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

    print(data_set_smiles)

    # Find all of the smiles which are common to the requested
    # data sets.
    common_smiles = None

    for smiles_set in data_set_smiles:

        if common_smiles is None:

            common_smiles = smiles_set
            continue

        common_smiles = common_smiles.intersection(smiles_set)

    logging.info(f'The combined sets have {len(common_smiles)} molecules in common.')
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
    allowed_elements = ['H', 'N', 'C', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Na', 'K', 'Ca']

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

        # Apply the high level filters.
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
        current_number_of_properties = data_set.number_of_properties

        logging.info(f'The filtered data set contains {data_set.number_of_properties} properties.')

        if property_type == DielectricConstant:

            # Filter out any measured dielectric constants which are too low.
            _filter_dielectric_constants(data_set, 10.0 * unit.dimensionless)

            logging.info(f'{current_number_of_properties - data_set.number_of_properties} '
                         f'dielectric properties had values less than 10.0 and were removed.')

        data_sets[(property_type, substance_type)] = data_set
        logging.info(f'Finished loading the {substance_type} {property_type} data set.')

        file_name = f'{property_type.__name__}_{str(substance_type.value)}.csv'
        file_path = os.path.join(filtered_data_sets_directory, file_name)

        PandasDataSet.to_pandas_csv(data_set, file_path)

    # Find those compounds for which there is data for all of the properties of
    # interest.
    #
    # TODO: Refactor the following to stepwise allow compounds common across all
    #       properties -> compounds common across some properties -> compounds
    #       not common across any properties until all smiles patterns are matched,
    #       or adding new compounds does not increase the SMIRKS coverage.
    #
    common_smiles = _find_common_smiles_patterns(
        *[data_sets[property_tuple] for property_tuple in properties_of_interest]
    )

    used_vdw_parameters = _find_smirks_parameters('vdW', *common_smiles)

    for smirks in used_vdw_parameters:

        if len(used_vdw_parameters[smirks]) == 0:
            continue

        print(f'{smirks} was exercised.')

    for smirks in used_vdw_parameters:

        if len(used_vdw_parameters[smirks]) > 0:
            continue

        print(f'{smirks} was not exercised.')

    # # Hide the overly verbose 'missing sterochemistry' toolkit logging.
    # logger = logging.getLogger()
    # logger.setLevel(logging.ERROR)
    #
    # # Find the set of smiles common to both the pure density and
    # # pure vapour pressure data sets.
    # common_smiles, data_counts = _find_common_smiles_patterns(
    #     # ('Density', 1),
    #     ('EnthalpyOfVapourisation', 1),
    #     # ('VaporPressure', 1),
    #     save_structure_pngs=False
    # )
    #
    # # Count the frequencies of smirks applied to pure systems.
    # # vdw_parameters_per_pure_substance = _count_parameters_per_molecule('vdW', *common_smiles)
    # # torsion_parameters_per_pure_substance = _count_parameters_per_molecule('ProperTorsions', *common_smiles)
    # #
    # # plt.title('VdW Parameters per Pure Substance')
    # # plt.bar(list(vdw_parameters_per_pure_substance.keys()),
    # #         list(vdw_parameters_per_pure_substance.values()))
    # # plt.show()
    # # plt.title('Torsion Parameters per Pure Substance')
    # # plt.bar(list(torsion_parameters_per_pure_substance.keys()),
    # #         list(torsion_parameters_per_pure_substance.values()))
    # # plt.show()
    # #
    # # print(f'VdW_Pure={vdw_parameters_per_pure_substance} '
    # #       f'ProperTorsions_Pure={torsion_parameters_per_pure_substance}')
    #
    # # Count the frequencies of smirks applied to binary systems.
    # # data_directory = get_data_filename('property_data')
    # # enthalpy_of_mixing_data_set = pandas.read_csv(os.path.join(data_directory, f'EnthalpyOfMixing_binary.csv'))
    # #
    # # vdw_parameters_per_binary_substance = defaultdict(int)
    # # torsion_parameters_per_binary_substance = defaultdict(int)
    # #
    # # unique_binary_pairs = set()
    # #
    # # for _, row in enthalpy_of_mixing_data_set.iterrows():
    # #     unique_binary_pairs.add((row['Component 1'], row['Component 2']))
    # #
    # # for smiles_0, smiles_1 in unique_binary_pairs:
    # #
    # #     smiles_by_vdw_smirks = _find_smirks_parameters('vdW', smiles_0, smiles_1)
    # #     total_vdw = len([1 for smirks in smiles_by_vdw_smirks if len(smiles_by_vdw_smirks[smirks]) > 0])
    # #     vdw_parameters_per_binary_substance[total_vdw] += 1
    # #
    # #     smiles_by_torsion_smirks = _find_smirks_parameters('ProperTorsions', smiles_0, smiles_1)
    # #     total_torsion = len([1 for smirks in smiles_by_torsion_smirks if len(smiles_by_torsion_smirks[smirks]) > 0])
    # #     torsion_parameters_per_binary_substance[total_torsion] += 1
    # #
    # # plt.title('VdW Parameters per Binary Substance')
    # # plt.bar(list(vdw_parameters_per_binary_substance.keys()),
    # #         list(vdw_parameters_per_binary_substance.values()))
    # # plt.show()
    # # plt.title('Torsion Parameters per Binary Substance')
    # # plt.bar(list(torsion_parameters_per_binary_substance.keys()),
    # #         list(torsion_parameters_per_binary_substance.values()))
    # # plt.show()
    # #
    # # print(f'VdW_Binary={vdw_parameters_per_binary_substance} '
    # #       f'ProperTorsions_Binary={torsion_parameters_per_binary_substance}')
    #
    # # Find the set of smiles common to both the pure density and
    # # pure enthalpy of vapourisation data sets.
    # # common_smiles, data_counts = _find_common_smiles_patterns(
    # #     ('Density', 1),
    # #     ('EnthalpyOfVapourisation', 1)
    # # )
    #
    # # Find the set of smiles common to both the pure and binary density,
    # # the pure static dielectric constant, the binary enthalpy of mixing,
    # # and the vapour pressure data sets.
    # # common_smiles, data_counts = _find_common_smiles_patterns(
    # #     ('Density', 1),
    # #     ('Density', 2),
    # #     ('DielectricConstant', 1),
    # #     ('EnthalpyOfMixing', 2),
    # #     ('VaporPressure', 1),
    # #     save_structure_pngs=False
    # # )
    #
    # # data_counts.to_csv('data_counts.csv')
    # #
    # # Find all of the vdw parameters which would be assigned to the common
    # # smiles patterns.
    # used_vdw_parameters = _find_smirks_parameters('vdW', *common_smiles)
    #
    # # Print information about those vdw parameters for which
    # # no matched smiles patterns were found.
    # for smirks in used_vdw_parameters:
    #
    #     if len(used_vdw_parameters[smirks]) == 0:
    #         continue
    #
    #     print(f'{smirks} was exercised.')
    #
    # for smirks in used_vdw_parameters:
    #
    #     if len(used_vdw_parameters[smirks]) > 0:
    #         continue
    #
    #     print(f'{smirks} was not exercised.')
    #
    # # Find all of the chemical moieties present in each of the
    # # common smiles patterns.
    # # chemical_moieties = analyse_functional_groups(common_smiles)


def _main():
    """A utility function for calling this script directly, which
    expects that the data files generated by the `parserawdata`
    script are located in the '~/property_data' directory.
    """

    home_directory = os.path.expanduser("~")
    data_directory = os.path.join(home_directory, 'property_data')

    curate_data_set(data_directory)


if __name__ == '__main__':
    _main()
