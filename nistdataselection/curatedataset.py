"""
Records the tools and decisions used to select NIST data for curation.
"""
import glob
import logging
import math
import os
import re
import shutil
import subprocess
from collections import defaultdict
from enum import Enum

import pandas
from openeye import oedepict
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines import smirnoff
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


def get_data_filename(relative_path: str):
    """Get the full path to one of the reference files in the
     data directory.

    Parameters
    ----------
    relative_path : str
        The relative path of the file to load.
    """

    from pkg_resources import resource_filename
    fn = resource_filename('nistdataselection', os.path.join('data', relative_path))

    if not os.path.exists(fn):
        raise FileNotFoundError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn


def cleanup_raw_data(raw_data_directory):
    """A helper function to strip all unpublished NIST values and uncertainties
    from the raw data, remove any duplicate, and produce count files.

    This function is mainly only for use by people extracting raw data
    directly from ThermoML archives using another tool.

    Parameters
    ----------
    raw_data_directory: str
        The file path to the raw data directory.
    """
    counts_directory = get_data_filename('property_counts')
    os.makedirs(counts_directory, exist_ok=True)

    data_directory = get_data_filename('property_data')
    os.makedirs(data_directory, exist_ok=True)

    for data_path in glob.glob(os.path.join(raw_data_directory, '*.csv')):

        print(f'Cleaning {data_path}.')
        raw_data = pandas.read_csv(data_path)

        if 'Uncertainty' in raw_data.columns:
            raw_data.drop(columns="Uncertainty", inplace=True)

        if 'Value' in raw_data.columns:
            raw_data.drop(columns="Value", inplace=True)

        raw_data = raw_data.drop_duplicates(['Temperature', 'Pressure', 'Substance', 'N_Components'])
        property_name = os.path.splitext(os.path.basename(data_path))[0]

        for index, data_type in enumerate(['pure', 'binary', 'ternary']):

            data = raw_data.loc[raw_data['N_Components'] == index + 1]
            print(f'Processing the {data_type} data.')

            substance_counts = {}

            column_names = ['Temperature', 'Pressure']

            for component_index in range(index + 1):
                column_names.append(f'Component {component_index + 1}')
                column_names.append(f'Mole Fraction {component_index + 1}')

            column_names.append('Source')

            refactored_data = pandas.DataFrame(columns=column_names)

            for _, data_row in data.iterrows():

                refactored_row = {
                    'Temperature': data_row['Temperature'],
                    'Pressure': data_row['Pressure'],
                    'Source': data_row['Source'],
                }

                substance_split = data_row['Substance'].split('|')

                for component_index in range(index + 1):

                    smiles = re.match('[^{]*', substance_split[component_index]).group(0)
                    mole_fraction = re.search(r'{(.*)}', substance_split[component_index]).group(1)

                    if smiles not in substance_counts:
                        substance_counts[smiles] = 0

                    substance_counts[smiles] += 1

                    refactored_row[f'Component {component_index + 1}'] = smiles
                    refactored_row[f'Mole Fraction {component_index + 1}'] = mole_fraction

                refactored_data = refactored_data.append(refactored_row, ignore_index=True)

            refactored_data.to_csv(os.path.join(data_directory, f'{property_name}_{data_type}.csv'), index=False)

            with open(os.path.join(counts_directory, f'{property_name}_{data_type}.csv'), 'w') as file:

                file.write(f'Smiles,Count\n')

                for smiles, count in substance_counts.items():
                    file.write(f'{smiles},{count}\n')

            print(f'Finished processing the {data_type} data.')

        print(f'Finished cleaning the {data_path} data.')


def find_smirks_parameters(parameter_tag='vdW', *smiles_patterns):
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

    force_field = smirnoff.ForceField(get_data_filename('smirnoff99Frosst-1.0.9.offxml'))
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

        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        topology = Topology.from_molecules([molecule])

        assigned_parameters = force_field.label_molecules(topology)[0]
        parameters_with_tag = assigned_parameters[parameter_tag]

        for parameter in parameters_with_tag.values():
            smiles_by_parameter_smirks[parameter.smirks].add(smiles)

    return smiles_by_parameter_smirks


def count_parameters_per_molecule(parameter_tag='vdW', *smiles_patterns):
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

    smiles_by_parameter_smirks = find_smirks_parameters(parameter_tag, *smiles_patterns)

    parameter_smirks_by_smiles = defaultdict(list)

    for smirks_pattern in smiles_by_parameter_smirks:
        for smiles_pattern in smiles_by_parameter_smirks[smirks_pattern]:
            parameter_smirks_by_smiles[smiles_pattern].append(smirks_pattern)

    counts = defaultdict(int)

    for smiles_pattern in parameter_smirks_by_smiles:

        number_of_parameters = len(parameter_smirks_by_smiles[smiles_pattern])
        counts[number_of_parameters] += 1

    return counts


def analyse_functional_groups(smiles_list=None):
    """Employs checkmol to determine which chemical moieties
    are present within a list of smiles patterns.

    Parameters
    ----------
    smiles_list: list of str, optional
        The smiles patterns to examine. If no list
        is provided, the data directory will be
        scanned for all smirks patterns.

    Returns
    -------
    dict of str and list of str
        A dictionary with keys of smiles patterns, and values
        of lists of matching chemical moiety descriptors.
    """

    # Make sure the checkmol utility has been installed separately.
    if shutil.which('checkmol') is None:

        raise FileNotFoundError('checkmol was not found on this machine. Visit '
                                'http://merian.pch.univie.ac.at/~nhaider/cheminf/cmmm.html '
                                'to obtain it.')

    # Find a list of smiles from the data files if none
    # is provided.
    if smiles_list is None:

        smiles_list = set()

        counts_directory = get_data_filename('property_counts')

        for counts_path in glob.glob(os.path.join(counts_directory, '*.csv')):

            property_counts = pandas.read_csv(counts_path)

            for smiles in property_counts['Smiles']:
                smiles_list.add(smiles)

    smiles_group_codes = {}
    molecule_file_name = 'tmp.sdf'

    for smiles in smiles_list:

        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        # Save the smile pattern out as an SDF file, ready
        # to use as input to checkmol.
        with open(molecule_file_name, 'w') as file:
            molecule.to_file(file, 'SDF')

        # Execute checkmol.
        result = subprocess.check_output(['checkmol', molecule_file_name],
                                         stderr=subprocess.STDOUT).decode()

        groups = []

        # Turn the string output into a list of moieties.
        if len(result) > 0:
            groups = list(filter(None, result.replace('\n', '').split(';')))

        smiles_group_codes[smiles] = groups

    # Remove the temporary SDF file.
    if os.path.isfile(molecule_file_name):
        os.unlink(molecule_file_name)

    return smiles_group_codes


def smiles_to_png(directory, smiles):
    """Creates a png image of the 2D representation of
    a given smiles pattern.

    Parameters
    ----------
    directory: str
        The directory to save the smiles pattern in.
    smiles: str
        The smiles pattern to generate the png of.
    """

    off_molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    oe_molecule = off_molecule.to_openeye()

    oedepict.OEPrepareDepiction(oe_molecule)

    options = oedepict.OE2DMolDisplayOptions(256, 256, oedepict.OEScale_AutoScale)

    display = oedepict.OE2DMolDisplay(oe_molecule, options)
    oedepict.OERenderMolecule(os.path.join(directory, f'{smiles}.png'), display)


def find_common_smiles_patterns(*properties_of_interest):
    """Find the set of smiles patterns which are common to multiple
    property data sets (and with different numbers of components).

    Parameters
    ----------
    properties_of_interest: list of tuple of str and int
        A list of descriptors of the property data sets to
        find the common smiles patterns between. The tuple
        should be of the form (property_name, SubstanceType)

    Returns
    -------
    set of str
        The smiles patterns which are common to all specified
        data sets.
    pandas.DataFrame
        A data from describing how many data points of each property
        there is for each common smiles pattern. The data from will
        have columns:

        Smiles,Pure/Binary/Ternary PropertyName, ..., Pure/Binary/Ternary PropertyName
    """

    assert len(properties_of_interest) > 0

    substance_type_to_int = {
        SubstanceType.Pure: 1,
        SubstanceType.Binary: 2,
        SubstanceType.Ternary: 3
    }

    data_directory = get_data_filename('property_data')

    data_sets = {}
    data_set_smiles = {}

    components_to_string = {1: 'pure', 2: 'binary', 3: 'ternary'}

    for property_name, number_of_components in properties_of_interest:

        if number_of_components > 3:
            raise ValueError('Only properties with up to three components are supported')

        component_string = components_to_string[number_of_components]

        if property_name not in data_sets:
            data_sets[property_name] = {}
            data_set_smiles[property_name] = {}

        # Load in the data set for this property.
        data_sets[property_name][number_of_components] = pandas.read_csv(
            os.path.join(data_directory, f'{property_name}_{component_string}.csv'))

        # Find all unique smiles in the data set.
        data_set_smiles[property_name][number_of_components] = set()

        for index in range(number_of_components):

            for smiles in data_sets[property_name][number_of_components][f'Component {index + 1}']:

                # Exclude any salt pairs.
                if '.' in smiles:
                    continue

                data_set_smiles[property_name][number_of_components].add(smiles)

    # Find all of the smiles which are common to the requested
    # data sets.
    common_smiles = None

    for property_name, number_of_components in properties_of_interest:

        if common_smiles is None:

            common_smiles = data_set_smiles[property_name][number_of_components]
            continue

        common_smiles = common_smiles.intersection(data_set_smiles[property_name][number_of_components])

    print(f'The combined sets have {len(common_smiles)} molecules in common.')

    # Collate the data counts into a pandas data frame.
    column_names = ['Smiles']

    # Set up the columns.
    for property_name, number_of_components in properties_of_interest:
        component_string = components_to_string[number_of_components]
        column_names.append(f'{component_string.capitalize()} {property_name.capitalize()}')

    data_counts = pandas.DataFrame(columns=column_names)

    # Add the row entries.
    for smiles in common_smiles:

        row = {'Smiles': smiles}

        for property_name, number_of_components in properties_of_interest:
            component_string = components_to_string[number_of_components]

            data_set = data_sets[property_name][number_of_components]

            matching_data = []

            if number_of_components == 1:
                matching_data = data_set.loc[data_set['Component 1'] == smiles]
            elif number_of_components == 2:
                matching_data = data_set.loc[(data_set['Component 1'] == smiles) |
                                             (data_set['Component 2'] == smiles)]
            elif number_of_components == 3:
                matching_data = data_set.loc[(data_set['Component 1'] == smiles) |
                                             (data_set['Component 2'] == smiles) |
                                             (data_set['Component 3'] == smiles)]

            row[f'{component_string.capitalize()} {property_name.capitalize()}'] = len(matching_data)

        data_counts = data_counts.append(row, ignore_index=True)

    return common_smiles, data_counts


def load_data_set(directory, properties_to_load):
    """

    Parameters
    ----------
    directory: str
        The path which contains the data csv files generated
        by the `parse_raw_data` method.
    properties_to_load: list of tuple
        A list of the properties to load from the data collection,
        where each entry is a tuple of the property of interest,
        and the substance type.

    Returns
    -------
    PhysicalPropertyDataSet
        The loaded data set.
    """

    assert os.path.isdir(directory)

    full_data_set = PhysicalPropertyDataSet()

    for property_type, substance_type in properties_to_load:

        # Try to load in the pandas data file.
        file_name = f'{property_type.__name__}_{str(substance_type.value)}.csv'
        file_path = os.path.join(directory, file_name)

        if not os.path.isfile(file_path):

            raise ValueError(f'No data file could be found for '
                             f'{substance_type} {property_type}s at {file_path}')

        data_set = PandasDataSet.from_pandas_csv(file_path, property_type)
        full_data_set.merge(data_set)

    return full_data_set


def remove_duplicates(data_set):
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
            pressure = physical_property.thermodynamic_state.temperature.value_in_unit(unit.kilopascal)

            state_tuple = (f'{temperature:.2f}', f'{pressure:.3f}')

            if state_tuple not in properties_by_substance[substance_id]:

                # Handle the easy case where this is the first time a
                # property at this state has been observed.
                properties_by_substance[substance_id][state_tuple] = physical_property
                continue

            existing_property = properties_by_substance[substance_id][state_tuple]

            existing_uncertainty = (math.inf if existing_property.uncertainty is None else
                                    existing_property.uncertainty)

            current_uncertainty = (math.inf if physical_property.uncertainty is None else
                                   physical_property.uncertainty)

            if (math.isinf(existing_uncertainty) and math.isinf(current_uncertainty) or
                existing_uncertainty < current_uncertainty):

                # If neither property has an uncertainty, or the existing one has
                # a lower uncertainty keep that one.
                continue

            properties_by_substance[substance_id][state_tuple] = physical_property

    # Rebuild the data set with only unique properties.
    unique_data_set = PhysicalPropertyDataSet()

    for substance_id in properties_by_substance:

        if substance_id not in unique_data_set.properties:
            unique_data_set.properties[substance_id] = []

        for property_type in properties_by_substance[substance_id]:

            for state_tuple in properties_by_substance[substance_id][property_type]:

                unique_data_set.properties[substance_id].append(
                    unique_data_set.properties[substance_id][property_type][state_tuple])

    logging.info(f'{unique_data_set.number_of_properties - data_set.number_of_properties} '
                 f'duplicate properties were removed.')

    return unique_data_set


def filter_dielectric_constants(data_set, minimum_value):
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

        return physical_property.value < minimum_value

    data_set.filter_by_function(filter_function)


def main():
    """The main function which will perform the
    data curation."""

    setup_timestamp_logging()

    home_directory = os.path.expanduser("~")
    property_data_directory = os.path.join(home_directory, 'property_data')

    # Define the properties which we are interested in curating data for,
    # as well as the types of data we are interested in.
    properties_of_interest = [
        (Density, SubstanceType.Pure),
        (DielectricConstant, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure)
    ]

    # Load the full data sets from the processed data files, and
    # remove any duplicate properties according to `remove_duplicates`.
    logging.info('Loading data sets.')

    data_set = load_data_set(property_data_directory, properties_of_interest)
    # Remove any duplicate properties.
    data_set = remove_duplicates(data_set)

    # Define the ranges of temperatures and pressures of interest.
    # Here we choose a range of temperatures which are biologically
    # relevant (15 C - 45 C) and pressures which are close to ambient.
    temperature_range = (288.15 * unit.kelvin, 318.15 * unit.kelvin)
    pressure_range = (0.95 * unit.atmosphere, 1.05 * unit.atmosphere)

    # Define the elements that we are interested in. Here we only allow
    # those elements for which smirnoff99Frosst has parameters for.
    allowed_elements = ['H', 'N', 'C', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Na', 'K', 'Ca']

    # Apply the high level filters.
    unfiltered_number_of_properties = data_set.number_of_properties

    logging.info('Filtering data sets.')
    data_set.filter_by_temperature(min_temperature=temperature_range[0],
                                   max_temperature=temperature_range[1])

    logging.info(f'{data_set.number_of_properties - unfiltered_number_of_properties} '
                 f'properties were outside of the temperature range and were removed.')

    data_set.filter_by_pressure(min_pressure=pressure_range[0],
                                max_pressure=pressure_range[1])

    logging.info(f'{data_set.number_of_properties - unfiltered_number_of_properties} '
                 f'properties were outside of the pressure range and were removed.')

    data_set.filter_by_elements(*allowed_elements)

    logging.info(f'{data_set.number_of_properties - unfiltered_number_of_properties} '
                 f'properties contained unwanted elements and were removed.')

    logging.info(f'The filtered data set contains {data_set.number_of_properties} properties.')

    # Filter out any measured dielectric constants which are too low.
    filter_dielectric_constants(data_set, 10.0 * unit.dimensionless)

    # Find those compounds for which there is data for all of the properties of
    # interest.
    #
    # TODO: Refactor the following to stepwise allow compounds common across all
    #       properties -> compounds common across some properties -> compounds
    #       not common across any properties until all smiles patterns are matched,
    #       or adding new compounds does not increase the SMIRKS coverage.
    #
    # common_smiles, data_counts = find_common_smiles_patterns(
    #     *properties_of_interest
    # )


    # # Hide the overly verbose 'missing sterochemistry' toolkit logging.
    # logger = logging.getLogger()
    # logger.setLevel(logging.ERROR)
    #
    # # Find the set of smiles common to both the pure density and
    # # pure vapour pressure data sets.
    # common_smiles, data_counts = find_common_smiles_patterns(
    #     # ('Density', 1),
    #     ('EnthalpyOfVapourisation', 1),
    #     # ('VaporPressure', 1),
    #     save_structure_pngs=False
    # )
    #
    # # Count the frequencies of smirks applied to pure systems.
    # # vdw_parameters_per_pure_substance = count_parameters_per_molecule('vdW', *common_smiles)
    # # torsion_parameters_per_pure_substance = count_parameters_per_molecule('ProperTorsions', *common_smiles)
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
    # #     smiles_by_vdw_smirks = find_smirks_parameters('vdW', smiles_0, smiles_1)
    # #     total_vdw = len([1 for smirks in smiles_by_vdw_smirks if len(smiles_by_vdw_smirks[smirks]) > 0])
    # #     vdw_parameters_per_binary_substance[total_vdw] += 1
    # #
    # #     smiles_by_torsion_smirks = find_smirks_parameters('ProperTorsions', smiles_0, smiles_1)
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
    # # common_smiles, data_counts = find_common_smiles_patterns(
    # #     ('Density', 1),
    # #     ('EnthalpyOfVapourisation', 1)
    # # )
    #
    # # Find the set of smiles common to both the pure and binary density,
    # # the pure static dielectric constant, the binary enthalpy of mixing,
    # # and the vapour pressure data sets.
    # # common_smiles, data_counts = find_common_smiles_patterns(
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
    # used_vdw_parameters = find_smirks_parameters('vdW', *common_smiles)
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


if __name__ == "__main__":
    main()
