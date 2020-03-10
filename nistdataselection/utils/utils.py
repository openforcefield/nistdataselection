"""Utilities to help with processing the NIST data sets.
"""
import functools
import logging
import math
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from enum import Enum

import cmiles.generator
from evaluator import unit
from evaluator.utils.openmm import openmm_quantity_to_pint
from openeye import oechem, oedepict
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils import UndefinedStereochemistryError

from nistdataselection.utils.pandas import data_frame_to_smiles_tuples

logger = logging.getLogger(__name__)


class SubstanceType(Enum):
    """An enum which encodes the names used for substances
    with different numbers of components.
    """

    Pure = "pure"
    Binary = "binary"
    Ternary = "ternary"


substance_type_to_int = {
    SubstanceType.Pure: 1,
    SubstanceType.Binary: 2,
    SubstanceType.Ternary: 3,
}

int_to_substance_type = {
    1: SubstanceType.Pure,
    2: SubstanceType.Binary,
    3: SubstanceType.Ternary,
}

cached_smirks_parameters = {}


def property_to_type_tuple(physical_property):
    """Converts a physical property into a tuple of it's
    type, and it's substance type.

    Parameters
    ----------
    physical_property: PhysicalProperty
        The physical property to tuplize.

    Returns
    -------
    tuple of type and SubstanceType
    """

    return (
        type(physical_property),
        int_to_substance_type[physical_property.substance.number_of_components],
    )


def property_to_snake_case(property_type):
    """Converts a property type to a snake case name.

    Parameters
    ----------
    property_type: type of PhysicalProperty of str
        The property type to convert.

    Returns
    -------
    str
        The property type as a snake case string.
    """

    if not isinstance(property_type, str):
        property_type = property_type.__name__

    return re.sub(r"(?<!^)(?=[A-Z])", "_", property_type).lower()


def property_to_title(property_type, substance_type, property_unit=None, latex_unit=False):
    """Converts a property type to a title case, e.g `ExcessMolarVolume`
    and `SubstanceType.Binary` becomes `Binary Excess Molar Volume`

    Parameters
    ----------
    property_type: type of PhysicalProperty of str
        The property type.
    substance_type: SubstanceType
        The substance type.
    property_unit: pint.Unit, optional
        The unit to include in the title.
    latex_unit: bool
        Whether or not to format the unit as a latex string.

    Returns
    -------
    str
        The property type as a snake case string.
    """

    if not isinstance(property_type, str):
        property_type = property_type.__name__

    property_name = " ".join(
        re.sub(
            "([A-Z][a-z]+)",
            r" \1",
            re.sub("([A-Z]+)", r" \1", property_type),
        ).split()
    )

    title = f"{substance_type.value} {property_name}".title()

    if property_unit is not None and property_unit != unit.dimensionless:

        if not latex_unit:
            title = f"{title} (${property_unit:~}$)"
        else:
            title = f"{title} (${property_unit:~L}$)"

    return title


def property_to_file_name(property_type, substance_type):
    """Converts a property type to a unified file name
    of {property_type}_{substance_type} where the property
    type is converted to snake case, and the substance type
    is lower case, e.g density_pure

    Parameters
    ----------
    property_type: type of PhysicalProperty of str
        The property type.
    substance_type: SubstanceType
        The substance type.

    Returns
    -------
    str
        The file name.
    """

    property_name = property_to_snake_case(property_type)
    file_name = f"{property_name}_{str(substance_type.value)}"

    return file_name


@functools.lru_cache(3000)
def get_atom_count(smiles):
    return Molecule.from_smiles(smiles, allow_undefined_stereo=True).n_atoms


@functools.lru_cache(3000)
def get_heavy_atom_count(smiles):

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    heavy_atoms = [atom for atom in molecule.atoms if atom.element.symbol != "H"]
    return len(heavy_atoms)


@functools.lru_cache(3000)
def get_molecular_weight(smiles):

    from simtk import unit as simtk_unit
    from openforcefield.topology import Molecule

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

    molecular_weight = 0.0 * simtk_unit.dalton

    for atom in molecule.atoms:
        molecular_weight += atom.mass

    return openmm_quantity_to_pint(molecular_weight)


@functools.lru_cache()
def _get_default_force_field():
    return ForceField("openff-1.0.0.offxml")


@functools.lru_cache(3000)
def find_smirks_matches(smirks_of_interest, *smiles_patterns):
    """Determines which of the specified smirks match the specified
    set of molecules.

    Parameters
    ----------
    smirks_of_interest: tuple of str
        The list of smirks to try and match against the given molecules.

    Returns
    -------
    dict of str and list of str
        A dictionary with keys of the smirks of interest, and
        values of lists of smiles patterns which match those
        smirks.
    """

    if len(smirks_of_interest) == 0:
        return {}

    matches_per_smiles = {}

    for smiles in smiles_patterns:

        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        matches_per_smiles[smiles] = [
            smirks
            for smirks in smirks_of_interest
            if len(molecule.chemical_environment_matches(smirks)) > 0
        ]

    matches_per_smirks = {smirks: set() for smirks in smirks_of_interest}

    for smiles, smirks_patterns in matches_per_smiles.items():

        for smirks in smirks_patterns:

            matches_per_smirks[smirks].add(smiles)

    return matches_per_smirks


@functools.lru_cache(3000)
def standardize_smiles(*smiles_patterns):

    return_values = []

    for smiles_pattern in smiles_patterns:

        identifiers = cmiles.generator.get_molecule_ids(smiles_pattern, strict=False)
        return_values.append(identifiers["canonical_smiles"])

    return return_values


def smiles_to_pdf(smiles, file_path, rows=10, columns=6):
    """Creates a PDF file containing images of a list of molecules
    described by their SMILES patterns.

    Parameters
    ----------
    smiles: list of str or tuple of str
        The SMILES patterns of the molecules. The list can either contain
        a list of single SMILES strings, or a tuple of SMILES strings. If
        tuples of SMILES are provided, these smiles will be grouped together
        in the output. All tuples in the list must have the same length.
    file_path: str
        The file path to save the pdf to.
    rows: int
        The maximum number of rows of molecules to include per page.
    columns: int
        The maximum number of molecules to include per row.
    """

    assert len(smiles) > 0

    # Validate the input type.
    assert all(isinstance(x, str) for x in smiles) or all(
        isinstance(x, tuple) for x in smiles
    )

    # Make sure the smiles tuples are the same length.
    molecules_per_group = 1

    if isinstance(smiles[0], tuple):

        assert (len(x) == len(smiles[0]) for x in smiles)
        molecules_per_group = len(smiles[0])

    # Convert the list of tuple to list of strings.
    if isinstance(smiles[0], tuple):
        smiles = [".".join(sorted(x)) for x in smiles]

    # Create OEMol objects for each unique smiles pattern provided.
    oe_molecules = {}

    unique_smiles = set(smiles)

    for smiles_pattern in unique_smiles:

        molecule = oechem.OEMol()
        oechem.OEParseSmiles(molecule, smiles_pattern)

        oe_molecules[smiles_pattern] = molecule

    # Take into account that each group may have more than one molecule
    columns = int(math.floor(columns / molecules_per_group))

    report_options = oedepict.OEReportOptions(rows, columns)
    report_options.SetHeaderHeight(25)
    report_options.SetFooterHeight(25)
    report_options.SetCellGap(4)
    report_options.SetPageMargins(10)

    report = oedepict.OEReport(report_options)

    cell_width, cell_height = report.GetCellWidth(), report.GetCellHeight()

    display_options = oedepict.OE2DMolDisplayOptions(
        cell_width, cell_height, oedepict.OEScale_Default * 0.5
    )
    display_options.SetAromaticStyle(oedepict.OEAromaticStyle_Circle)

    pen = oedepict.OEPen(oechem.OEBlack, oechem.OEBlack, oedepict.OEFill_On, 1.0)
    display_options.SetDefaultBondPen(pen)

    interface = oechem.OEInterface()
    oedepict.OESetup2DMolDisplayOptions(display_options, interface)

    for i, smiles_pattern in enumerate(smiles):

        cell = report.NewCell()

        oe_molecule = oechem.OEMol(oe_molecules[smiles_pattern])
        oedepict.OEPrepareDepiction(oe_molecule, False, True)

        display = oedepict.OE2DMolDisplay(oe_molecule, display_options)
        oedepict.OERenderMolecule(cell, display)

    oedepict.OEWriteReport(file_path, report)


def data_frame_to_pdf(data_frame, file_path, rows=10, columns=6):
    """Creates a PDF file containing images of a the of substances
    contained in a data frame.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame containing the different substances.
    file_path: str
        The file path to save the pdf to.
    rows: int
        The maximum number of rows of molecules to include per page.
    columns: int
        The maximum number of molecules to include per row.
    """

    if len(data_frame) == 0:
        return

    smiles_tuples = data_frame_to_smiles_tuples(data_frame)
    smiles_to_pdf(smiles_tuples, file_path, rows, columns)


def find_parameter_smirks_matches(parameter_tag="vdW", *smiles_patterns):
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

    force_field = _get_default_force_field()
    parameter_handler = force_field.get_parameter_handler(parameter_tag)

    # Initialize the array with all possible smirks pattern
    # to make it easier to identify which are missing.
    smiles_by_parameter_smirks = {
        parameter.smirks: set() for parameter in parameter_handler.parameters
    }

    # Populate the dictionary using the open force field toolkit.
    for smiles in smiles_patterns:

        if (
            smiles not in cached_smirks_parameters
            or parameter_tag not in cached_smirks_parameters[smiles]
        ):

            try:
                molecule = Molecule.from_smiles(smiles)
            except UndefinedStereochemistryError:
                # Skip molecules with undefined stereochemistry.
                continue

            topology = Topology.from_molecules([molecule])

            if smiles not in cached_smirks_parameters:
                cached_smirks_parameters[smiles] = {}

            if parameter_tag not in cached_smirks_parameters[smiles]:
                cached_smirks_parameters[smiles][parameter_tag] = []

            cached_smirks_parameters[smiles][parameter_tag] = [
                parameter.smirks
                for parameter in force_field.label_molecules(topology)[0][
                    parameter_tag
                ].values()
            ]

        parameters_with_tag = cached_smirks_parameters[smiles][parameter_tag]

        for smirks in parameters_with_tag:
            smiles_by_parameter_smirks[smirks].add(smiles)

    return smiles_by_parameter_smirks


def invert_dict_of_iterable(dictionary, iterable_type=list):
    """Inverts a dictionary of string keys with values of
    lists of strings.

    Parameters
    ----------
    dictionary: dict of str and iterable of str
        The dictionary to invert
    iterable_type: type
        The type of iterable to use inv the inverted dictionary.

    Returns
    -------
    dictionary: dict of str and list of `iterable_type`
        The inverted dictionary
    """

    inverted_dictionary = defaultdict(iterable_type)

    for key in dictionary:
        for list_value in dictionary[key]:

            inverted_dictionary[list_value].append(key)

    return inverted_dictionary


def invert_dict_of_list(dictionary):
    """Inverts a dictionary of string keys with values of
    lists of strings.

    Parameters
    ----------
    dictionary: dict of str and list of str
        The dictionary to invert

    Returns
    -------
    dictionary: dict of str and list of str
        The inverted dictionary
    """
    return invert_dict_of_iterable(dictionary, list)


@functools.lru_cache(3000)
def analyse_functional_groups(smiles):
    """Employs checkmol to determine which chemical moieties
    are encoded by a given smiles pattern.

    Notes
    -----
    See https://homepage.univie.ac.at/norbert.haider/cheminf/fgtable.pdf
    for information about the group numbers (i.e moiety types).

    Parameters
    ----------
    smiles: str
        The smiles pattern to examine.
    Returns
    -------
    dict of str and int, optional
        A dictionary where each key corresponds to the `checkmol` defined group
        number, and each value if the number of instances of that moiety. If
        `checkmol` did not execute correctly, returns None.
    """

    # Make sure the checkmol utility has been installed separately.
    if shutil.which("checkmol") is None:

        raise FileNotFoundError(
            "checkmol was not found on this machine. Visit http://merian.pch.univie.ac."
            "at/~nhaider/cheminf/cmmm.html to obtain it."
        )

    oe_molecule = cmiles.utils.load_molecule(smiles, toolkit="openeye")

    # Save the smile pattern out as an SDF file, ready to use as input to checkmol.
    with tempfile.NamedTemporaryFile(suffix=".sdf") as file:

        output_stream = oechem.oemolostream(file.name)
        output_stream.SetFormat(oechem.OEFormat_SDF)

        oechem.OEWriteMolecule(output_stream, oe_molecule)

        # Execute checkmol.
        try:
            result = subprocess.check_output(
                ["checkmol", "-p", file.name], stderr=subprocess.STDOUT,
            ).decode()

        except subprocess.CalledProcessError:
            logger.exception("An exception was raised while calling checkmol.")
            result = ""

    if len(result) == 0:
        return None

    groups = {}

    for group in result.splitlines():

        group_code, group_count, _ = group.split(":")
        groups[group_code[1:]] = int(group_count)

    return groups
