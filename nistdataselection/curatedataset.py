"""
Records the tools and decisions used to select NIST data for curation.
"""
import glob
import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict

import pandas
from openeye import oedepict
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines import smirnoff


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


def find_common_smiles_patterns(*properties_of_interest, save_structure_pngs=True):
    """Find the set of smiles patterns which are common to multiple
    property data sets (and with different numbers of components).

    Parameters
    ----------
    properties_of_interest: list of tuple of str and int
        A list of descriptors of the property data sets to
        find the common smiles patterns between. The tuple
        should be of the form (property_name, number_of_components)
    save_structure_pngs: bool
        If true, a png image of the 2D representation of
        each of the common smiles pattern is saved to a
        figures directory.

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

    # Create png representations of the molecules if requested.
    if save_structure_pngs:
        os.makedirs('figures', exist_ok=True)

        for smiles in common_smiles:
            smiles_to_png('figures', smiles)

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


def main():
    """The main function which will perform the
    data curation."""

    # Hide the overly verbose 'missing sterochemistry' toolkit logging.
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    # Find the set of smiles common to both the pure density and
    # pure vapour pressure data sets.
    common_smiles, data_counts = find_common_smiles_patterns(
        # ('Density', 1),
        ('EnthalpyOfVapourisation', 1),
        # ('VaporPressure', 1),
        save_structure_pngs=False
    )

    # Count the frequencies of smirks applied to pure systems.
    # vdw_parameters_per_pure_substance = count_parameters_per_molecule('vdW', *common_smiles)
    # torsion_parameters_per_pure_substance = count_parameters_per_molecule('ProperTorsions', *common_smiles)
    #
    # plt.title('VdW Parameters per Pure Substance')
    # plt.bar(list(vdw_parameters_per_pure_substance.keys()),
    #         list(vdw_parameters_per_pure_substance.values()))
    # plt.show()
    # plt.title('Torsion Parameters per Pure Substance')
    # plt.bar(list(torsion_parameters_per_pure_substance.keys()),
    #         list(torsion_parameters_per_pure_substance.values()))
    # plt.show()
    #
    # print(f'VdW_Pure={vdw_parameters_per_pure_substance} '
    #       f'ProperTorsions_Pure={torsion_parameters_per_pure_substance}')

    # Count the frequencies of smirks applied to binary systems.
    # data_directory = get_data_filename('property_data')
    # enthalpy_of_mixing_data_set = pandas.read_csv(os.path.join(data_directory, f'EnthalpyOfMixing_binary.csv'))
    #
    # vdw_parameters_per_binary_substance = defaultdict(int)
    # torsion_parameters_per_binary_substance = defaultdict(int)
    #
    # unique_binary_pairs = set()
    #
    # for _, row in enthalpy_of_mixing_data_set.iterrows():
    #     unique_binary_pairs.add((row['Component 1'], row['Component 2']))
    #
    # for smiles_0, smiles_1 in unique_binary_pairs:
    #
    #     smiles_by_vdw_smirks = find_smirks_parameters('vdW', smiles_0, smiles_1)
    #     total_vdw = len([1 for smirks in smiles_by_vdw_smirks if len(smiles_by_vdw_smirks[smirks]) > 0])
    #     vdw_parameters_per_binary_substance[total_vdw] += 1
    #
    #     smiles_by_torsion_smirks = find_smirks_parameters('ProperTorsions', smiles_0, smiles_1)
    #     total_torsion = len([1 for smirks in smiles_by_torsion_smirks if len(smiles_by_torsion_smirks[smirks]) > 0])
    #     torsion_parameters_per_binary_substance[total_torsion] += 1
    #
    # plt.title('VdW Parameters per Binary Substance')
    # plt.bar(list(vdw_parameters_per_binary_substance.keys()),
    #         list(vdw_parameters_per_binary_substance.values()))
    # plt.show()
    # plt.title('Torsion Parameters per Binary Substance')
    # plt.bar(list(torsion_parameters_per_binary_substance.keys()),
    #         list(torsion_parameters_per_binary_substance.values()))
    # plt.show()
    #
    # print(f'VdW_Binary={vdw_parameters_per_binary_substance} '
    #       f'ProperTorsions_Binary={torsion_parameters_per_binary_substance}')

    # Find the set of smiles common to both the pure density and
    # pure enthalpy of vapourisation data sets.
    # common_smiles, data_counts = find_common_smiles_patterns(
    #     ('Density', 1),
    #     ('EnthalpyOfVapourisation', 1)
    # )

    # Find the set of smiles common to both the pure and binary density,
    # the pure static dielectric constant, the binary enthalpy of mixing,
    # and the vapour pressure data sets.
    # common_smiles, data_counts = find_common_smiles_patterns(
    #     ('Density', 1),
    #     ('Density', 2),
    #     ('DielectricConstant', 1),
    #     ('EnthalpyOfMixing', 2),
    #     ('VaporPressure', 1),
    #     save_structure_pngs=False
    # )

    # data_counts.to_csv('data_counts.csv')
    #
    # Find all of the vdw parameters which would be assigned to the common
    # smiles patterns.
    used_vdw_parameters = find_smirks_parameters('vdW', *common_smiles)

    # Print information about those vdw parameters for which
    # no matched smiles patterns were found.
    for smirks in used_vdw_parameters:

        if len(used_vdw_parameters[smirks]) == 0:
            continue

        print(f'{smirks} was exercised.')

    for smirks in used_vdw_parameters:

        if len(used_vdw_parameters[smirks]) > 0:
            continue

        print(f'{smirks} was not exercised.')

    # Find all of the chemical moieties present in each of the
    # common smiles patterns.
    # chemical_moieties = analyse_functional_groups(common_smiles)


if __name__ == "__main__":
    main()
