"""
Records the tools and decisions used to select NIST data for curation.
"""
import os
import re
import shutil
import subprocess
from collections import defaultdict

import pandas
from propertyestimator.client import PropertyEstimatorOptions
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.layers import SimulationLayer
from propertyestimator.properties import Density, DielectricConstant, EnthalpyOfVaporization, ExcessMolarVolume, \
    MeasurementSource, EnthalpyOfMixing
from propertyestimator.protocols.groups import ConditionalGroup
from propertyestimator.utils import setup_timestamp_logging
from propertyestimator.workflow import WorkflowOptions
from tabulate import tabulate

from nistdataselection.utils import PandasDataSet
from nistdataselection.utils.utils import smiles_to_png, cached_smirks_parameters, find_smirks_parameters, \
    int_to_substance_type, substance_type_to_int, invert_dict_of_list


def _estimate_required_simulations(properties_of_interest, data_set):
    """Estimate how many simulations the property estimator
    will try and run to estimate the given data set of properties.

    Parameters
    ----------
    properties_of_interest: list of tuple of type and SubstanceType
        A list of the property types which are of interest to optimise against.
    data_set: PhysicalPropertyDataSet
        The data set containing the data set of properties of interest.

    Returns
    -------
    int
        The estimated number of simulations required.
    """

    data_set = PhysicalPropertyDataSet.parse_json(data_set.json())

    options = PropertyEstimatorOptions()
    calculation_layer = 'SimulationLayer'

    for property_type, _ in properties_of_interest:

        options.workflow_options[property_type.__name__] = {calculation_layer: WorkflowOptions()}

        default_schema = property_type.get_default_workflow_schema(calculation_layer, WorkflowOptions())
        options.workflow_schemas[property_type.__name__] = {calculation_layer: default_schema}

    properties = []

    for substance_id in data_set.properties:
        properties.extend(data_set.properties[substance_id])

    workflow_graph = SimulationLayer._build_workflow_graph('', properties, '', [], options)

    number_of_simulations = 0

    for protocol_id in workflow_graph._protocols_by_id:

        protocol = workflow_graph._protocols_by_id[protocol_id]

        if not isinstance(protocol, ConditionalGroup):
            continue

        number_of_simulations += 1

    return number_of_simulations


def _sanitize_identifier(identifier_pattern):

    identifier_pattern = identifier_pattern.replace('\\', '\\\\')
    identifier_pattern = identifier_pattern.replace('#', '\\#')

    escaped_string = f'\\seqsplit{{{identifier_pattern}}}'
    escaped_string.replace('~', r'\textasciitilde')

    return escaped_string


def _property_tuple_to_string(property_type, substance_type):

    property_name = ' '.join(re.sub('([A-Z][a-z]+)', r' \1',
                                    re.sub('([A-Z]+)', r' \1', property_type.__name__)).split())

    return f'{str(substance_type.value).title()} {property_name.title()}'


def _property_tuple_to_latex_symbol(property_type, substance_type):

    property_type_to_symbol = {
        Density: r'$\rho$',
        DielectricConstant: r'$\epsilon_0$',
        EnthalpyOfVaporization: r'$\Delta H_{vap}$',
        EnthalpyOfMixing: r'$H_{mix}$',
        ExcessMolarVolume: r'$\rho_{excess}$',
    }

    return f'{str(substance_type.value).title()} {property_type_to_symbol[property_type]}'


def _write_header(margin_size_cm=3):

    return '\n'.join([
        r'\documentclass{article}',
        f'\\usepackage[margin={margin_size_cm}cm]{{geometry}}',
        '',
        r'\usepackage[utf8]{inputenc}',
        r'\usepackage{graphicx}',
        r'\usepackage{array}',
        r'\usepackage[export]{adjustbox}',
        r'\usepackage{parskip}',
        '',
        r'\usepackage{amssymb}',
        r'\usepackage{seqsplit}',
        '',
        r'\usepackage{url}',
        r'\urlstyle{same}',
        '',
        r'\usepackage{subfigure}',
        r'\usepackage{alphalph}',
        r'\renewcommand*{\thesubfigure}{%',
        r'\alphalph{\value{subfigure}}%',
        r'}%',
        '',
        r'\edef\hash {\string#}',
        '',
        r'\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}'
        '',
        r'\begin{document}',
    ])


def _write_title(number_of_substances, number_of_properties, number_of_simulations):

    return '\n'.join([
        r'\begin{center}',
        r'    \LARGE{Chosen Data Set}',
        r'    \vspace{.2cm}',
        r'    \large{https://github.com/openforcefield/nistdataselection}',
        r'\end{center}',
        '',
        f'A total of {number_of_properties} data points covering '
        f'{number_of_substances} unique molecules are to be optimized against. '
        f'This will require approximately {number_of_simulations} unique simulation to be '
        f'performed.',
    ])


def _write_smirks_exercised_table(property_tuples, all_vdw_smirks, data_points_per_vdw_smirks):

    columns = ['VdW SMIRKS']
    columns.extend([_property_tuple_to_latex_symbol(*property_tuple) for property_tuple in property_tuples])

    rows = []

    for smirks in all_vdw_smirks:

        safe_smirks = _sanitize_identifier(smirks)
        row = {'VdW SMIRKS': f"'{safe_smirks}'"}

        for property_tuple in property_tuples:

            property_string = _property_tuple_to_latex_symbol(*property_tuple)
            row[property_string] = data_points_per_vdw_smirks[smirks][property_tuple]

        rows.append(row)

    data_frame = pandas.DataFrame(data=rows, columns=columns)
    data_frame.sort_values(columns[1:], ascending=False, inplace=True)

    table_string_split = tabulate(data_frame, headers='keys', tablefmt='latex_raw', showindex=False).split('\n')
    table_string_split = table_string_split[1:]

    smirks_width = 13.5 - 1.25 * (len(columns) - 1)
    header_string = f'{{m{{{smirks_width}cm}} ' + ' '.join(['C{1.25cm}' for _ in range(len(columns) - 1)]) + '}'

    table_string = '\n'.join([
        f'\\begin{{tabular}}{header_string}',
        *table_string_split
    ])

    table_string = '\n'.join([
        '',
        r'\vspace{.3cm}',
        r'\begin{center}',
        f'    \\large{{\\textbf{{Data Points Per VdW SMIRKS Pattern}}}}',
        r'\end{center}'
        r'\vspace{.3cm}',
        ''
        r'\vspace{.3cm}',
        table_string,
        r'\vspace{.3cm}'])

    table_string = table_string.replace('\'\\', '\\')
    table_string = table_string.replace('}\'', '}')

    return table_string


def _write_unique_substances_per_property_table(property_tuples, data_count_per_substance):

    columns = [_property_tuple_to_latex_symbol(*property_tuple) for property_tuple in property_tuples]

    smiles_tuples_per_property = {property_tuple: set() for property_tuple in property_tuples}

    for substance in data_count_per_substance:

        substance_smiles = [component.smiles for component in substance.components]
        smiles_tuple = tuple(sorted(substance_smiles))

        for property_tuple in data_count_per_substance[substance]:
            smiles_tuples_per_property[property_tuple].add(smiles_tuple)

    row = {}

    for property_tuple in smiles_tuples_per_property:

        property_string = _property_tuple_to_latex_symbol(*property_tuple)
        row[property_string] = len(smiles_tuples_per_property[property_tuple])

    data_frame = pandas.DataFrame(data=[row], columns=columns)
    data_frame.sort_values(columns[1:], ascending=False, inplace=True)

    table_string = '\n'.join([
        '',
        r'\vspace{.3cm}',
        r'\begin{center}',
        f'    \\large{{\\textbf{{Unique Substances Per Data Type}}}}',
        r'\end{center}'
        r'\vspace{.3cm}',
        ''
        r'\vspace{.3cm}',
        tabulate(data_frame, headers='keys', tablefmt='latex_raw', showindex=False),
        r'\vspace{.3cm}'])

    table_string = table_string.replace('\'\\', '\\')
    table_string = table_string.replace('}\'', '}')

    return table_string


def _write_smiles_section(smiles_tuple, exercised_vdw_smirks_patterns, full_data_set, property_tuples):

    smiles_header = ' + '.join([_sanitize_identifier(smiles_pattern) for smiles_pattern in smiles_tuple])

    row_template = [
        r'\newpage',
        '',
        r'\hrulefill',
        '',
        r'\vspace{.3cm}',
        r'\begin{center}',
        f'    \\large{{\\textbf{{{smiles_header}}}}}',
        r'\end{center}'
        r'\vspace{.3cm}',
        ''
    ]

    for smiles_pattern in smiles_tuple:

        exercised_smirks = [smirks for smirks in exercised_vdw_smirks_patterns if
                            smiles_pattern in exercised_vdw_smirks_patterns[smirks]]

        exercised_smirks_strings = [f'\\item {{{_sanitize_identifier(smirks)}}}' for smirks in exercised_smirks]

        image_file_name = smiles_pattern.replace('/', '').replace('\\', '')

        row_template.extend([
            r'\begin{tabular}{ m{5cm} m{9cm} }',
            '    {Structure} & {SMIRKS Exercised} \\\\',
            f'    {{\\catcode`\\#=12 \\includegraphics{{{"./images/" + image_file_name + ".png"}}}}} & '
            f'\\begin{{itemize}} {" ".join(exercised_smirks_strings)} \\end{{itemize}} \\\\',
            r'\end{tabular}'
        ])

    for property_type, substance_type in property_tuples:

        def filter_by_substance_type(property_to_filter):
            return substance_type_to_int[substance_type] == len(property_to_filter.substance.components)

        def filter_by_smiles_tuple(property_to_filter):

            smiles_list = list(smiles_tuple)

            for component in property_to_filter.substance.components:

                if component.smiles not in smiles_list:
                    return False

                smiles_list.remove(component.smiles)

            return len(smiles_list) == 0

        data_set = PhysicalPropertyDataSet.parse_json(full_data_set.json())
        data_set.filter_by_property_types(property_type)
        data_set.filter_by_function(filter_by_substance_type)
        data_set.filter_by_function(filter_by_smiles_tuple)

        for substance_id in data_set.properties:

            for physical_property in data_set.properties[substance_id]:

                if len(physical_property.source.doi) > 0:
                    continue

                physical_property.source = MeasurementSource(
                    reference=os.path.basename(physical_property.source.reference))

        pandas_data_frame = PandasDataSet.to_pandas_data_frame(data_set)

        if pandas_data_frame.shape[0] == 0:
            continue

        headers_to_keep = ['Temperature (K)', 'Pressure (kPa)']
        header_to_sort = ['Pressure (kPa)', 'Temperature (K)']

        mole_fraction_index = 0

        while f'Mole Fraction {mole_fraction_index + 1}' in pandas_data_frame:

            headers_to_keep.append(f'Mole Fraction {mole_fraction_index + 1}')
            header_to_sort.append(f'Mole Fraction {mole_fraction_index + 1}')
            mole_fraction_index += 1

        headers_to_keep.append('Source')

        pandas_data_frame = pandas_data_frame[headers_to_keep]
        pandas_data_frame = pandas_data_frame.sort_values(header_to_sort)

        property_name = ' '.join(re.sub('([A-Z][a-z]+)', r' \1',
                                        re.sub('([A-Z]+)', r' \1', property_type.__name__)).split())

        row_template.append(f'\n{str(substance_type.value).title()} {property_name.title()} Data\n')
        row_template.append('\\vspace{.3cm}\n')
        row_template.append(tabulate(pandas_data_frame, headers='keys', tablefmt='latex', showindex=False))
        row_template.append('\\vspace{.3cm}\n')

    return '\n\n'.join(row_template) + '\n'


def _write_substances_per_data_type_section(property_tuple, data_count_per_substance, total_molecules_per_row):

    smiles_tuples = set()

    for substance in data_count_per_substance:

        if property_tuple not in data_count_per_substance[substance]:
            continue

        substance_smiles = [component.smiles for component in substance.components]
        smiles_tuples.add(tuple(sorted(substance_smiles)))

    property_string = _property_tuple_to_latex_symbol(*property_tuple)

    row_template = [
        '',
        r'\newpage',
        '',
        r'\hrulefill',
        '',
        r'\vspace{.3cm}',
        r'\begin{center}',
        f'    \\large{{\\textbf{{{property_string}}}}}',
        r'\end{center}'
        r'\vspace{.3cm}',
        '',
        r'\begin{figure}[h!]%',
    ]

    image_width_fraction = 1.0 / total_molecules_per_row * 0.98
    line_counter = 0

    for smiles_tuple in smiles_tuples:

        line_counter += len(smiles_tuple)

        if line_counter > total_molecules_per_row:
            row_template.extend([r'\end{figure}', '', r'\begin{figure}[h!]%'])
            line_counter = len(smiles_tuple)

        subfigure = [
            f'\\subfigure[][]{{%',
        ]

        for smiles_pattern in smiles_tuple:

            image_file_name = smiles_pattern.replace('/', '').replace('\\', '')
            image_file_path = './images/' + image_file_name + '.png'
            image_file_path = image_file_path.replace('#', '\\hash ')

            subfigure.append(f'\\includegraphics[width={image_width_fraction}\\textwidth]'
                             f'{{{image_file_path}}}%')

        subfigure[-1] = subfigure[-1][:-1] + r'}%'

        row_template.extend(subfigure)

    row_template.append(r'\end{figure}')

    return row_template


def _write_substances_per_data_type_sections(property_tuples, data_count_per_substance, total_molecules_per_row=8):

    all_sections = []

    for property_tuple in property_tuples:

        all_sections.append('\n'.join(_write_substances_per_data_type_section(property_tuple,
                                                                              data_count_per_substance,
                                                                              total_molecules_per_row)))

    return '\n'.join(all_sections)


def _create_molecule_images(chosen_smiles, directory):
    """Creates a PNG image of the 2D representation of each
    molecule represented in a list of smiles patterns.

    Parameters
    ----------
    chosen_smiles: list of str
        The list of molecule smiles representations.
    directory: str
        The directory to save the created images in.
    """
    os.makedirs(directory, exist_ok=True)

    for smiles in chosen_smiles:

        file_name = smiles.replace('/', '').replace('\\', '')
        file_path = os.path.join(directory, f'{file_name}.png')

        smiles_to_png(smiles, file_path)


def generate_report(data_set_path='curated_data_set.json', report_name='report', vdw_smirks_of_interest=None):
    """A helper utility which will take as input a PhysicalPropertyDataSet
    and generate a report of its contents and coverage.

    Parameters
    ----------
    data_set_path: str
        The path to the data set.
    report_name: str
        The name of the report files to generate.
    vdw_smirks_of_interest: list of str, optional
        The vdW smirks patterns which should be included in the
        summary table. If `None`, all vdW smirks will be included.
    """

    setup_timestamp_logging()

    with open(data_set_path) as file:
        data_set = PhysicalPropertyDataSet.parse_json(file.read())

    all_substances = set()

    all_smiles = set()
    all_smiles_tuples = set()

    all_property_types = set()

    data_count_per_substance = defaultdict(lambda: defaultdict(int))
    data_per_substance = defaultdict(lambda: defaultdict(list))

    for substance_id in data_set.properties:

        if len(data_set.properties[substance_id]) == 0:
            continue

        for physical_property in data_set.properties[substance_id]:

            substance_type = int_to_substance_type[physical_property.substance.number_of_components]
            property_type_tuple = (type(physical_property), substance_type)

            all_property_types.add(property_type_tuple)
            all_substances.add(physical_property.substance)

            for component in physical_property.substance.components:
                all_smiles.add(component.smiles)

            all_smiles_tuples.add(tuple(sorted([component.smiles for component in
                                                physical_property.substance.components])))

            data_count_per_substance[physical_property.substance][property_type_tuple] += 1
            data_per_substance[physical_property.substance][property_type_tuple].append(physical_property)

    # Determine the number of unique molecules
    number_of_substances = len(all_smiles)

    # Determine the list of all exercised vdW smirks patterns.
    all_vdw_smirks_patterns = vdw_smirks_of_interest

    if all_vdw_smirks_patterns is None:
        all_vdw_smirks_patterns = [smirks for smirks in find_smirks_parameters('vdW').keys()]

    exercised_vdw_smirks_patterns = find_smirks_parameters('vdW', *all_smiles)

    # Invert the exercised_vdw_smirks_patterns dictionary.
    vdw_smirks_patterns_by_smiles = invert_dict_of_list(exercised_vdw_smirks_patterns)

    # Count the number of data points per smirks pattern.
    data_points_per_vdw_smirks = defaultdict(lambda: defaultdict(int))

    for substance in data_count_per_substance:

        exercised_smirks = set()

        for component in substance.components:
            exercised_smirks.update(vdw_smirks_patterns_by_smiles[component.smiles])

        for smirks in exercised_smirks:

            if smirks not in all_vdw_smirks_patterns:
                continue

            for data_tuple in data_count_per_substance[substance]:
                data_points_per_vdw_smirks[smirks][data_tuple] += 1

    number_of_simulations = _estimate_required_simulations(all_property_types, data_set)

    _create_molecule_images(all_smiles, 'images')

    smiles_sections = '\n'.join([_write_smiles_section(smiles_tuple, exercised_vdw_smirks_patterns, data_set,
                                                       all_property_types) for smiles_tuple in all_smiles_tuples])

    latex_document = '\n\n'.join([
        _write_header(),
        _write_title(number_of_substances, data_set.number_of_properties, number_of_simulations),
        _write_smirks_exercised_table(all_property_types, all_vdw_smirks_patterns, data_points_per_vdw_smirks),
        _write_unique_substances_per_property_table(all_property_types, data_count_per_substance),
        _write_substances_per_data_type_sections(all_property_types, data_count_per_substance),
        r'\pagebreak',
        smiles_sections,
        r'\end{document}'
    ])

    report_path = report_name + '.tex'

    with open(report_path, 'w') as file:
        file.write(latex_document)

    if shutil.which('pdflatex') is not None:

        subprocess.call(['pdflatex', '-synctex=1', '-interaction=nonstopmode', report_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)


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

    data_set_path = 'curated_data_set.json'

    vdw_smirks_of_interest = [
        '[#1:1]-[#6X4]',
        '[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]',
        '[#1:1]-[#6X3]',
        '[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]',
        '[#1:1]-[#8]',
        '[#6:1]',
        '[#6X4:1]',
        '[#8:1]',
        '[#8X2H0+0:1]',
        '[#8X2H1+0:1]',
        '[#7:1]',
        '[#16:1]',
        '[#9:1]',
        '[#17:1]',
        '[#35:1]'
    ]

    generate_report(data_set_path, 'report', vdw_smirks_of_interest)

    # Cache the smirks which will be assigned to the different molecules
    # to speed up future runs.
    with open(cached_smirks_file_name, 'w') as file:
        json.dump(cached_smirks_parameters, file)


if __name__ == '__main__':
    _main()