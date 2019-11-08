"""Utilities to help with processing the NIST data sets.
"""
import functools
import logging
from collections import defaultdict
from enum import Enum

from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils import UndefinedStereochemistryError


class SubstanceType(Enum):
    """An enum which encodes the names used for substances
    with different numbers of components.
    """

    Pure = "pure"
    Binary = "binary"
    Ternary = "ternary"


substance_type_to_int = {SubstanceType.Pure: 1, SubstanceType.Binary: 2, SubstanceType.Ternary: 3}

int_to_substance_type = {1: SubstanceType.Pure, 2: SubstanceType.Binary, 3: SubstanceType.Ternary}

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


@functools.lru_cache(1000)
def get_atom_count(smiles):
    return Molecule.from_smiles(smiles).n_atoms


@functools.lru_cache()
def _get_default_force_field():
    return ForceField("openff-1.0.0.offxml")


def find_smirks_matches(smirks_of_interest, *smiles_patterns):
    """Determines which of the specified smirks match the specified
    set of molecules.

    Parameters
    ----------
    smirks_of_interest: list of str
        The list of smirks to try and match against the given molecules.

    Returns
    -------
    dict of str and list of str
        A dictionary with keys of the smirks of interest, and
        values of lists of smiles patterns which match those
        smirks.
    """
    matches_per_smiles = {}

    for smiles in smiles_patterns:

        molecule = Molecule.from_smiles(smiles)

        matches_per_smiles[smiles] = [
            smirks for smirks in smirks_of_interest if len(molecule.chemical_environment_matches(smirks)) > 0
        ]

    matches_per_smirks = {smirks: set() for smirks in smirks_of_interest}

    for smiles, smirks_patterns in matches_per_smiles.items():

        for smirks in smirks_patterns:

            matches_per_smirks[smirks].add(smiles)

    return matches_per_smirks


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
    smiles_by_parameter_smirks = {parameter.smirks: set() for parameter in parameter_handler.parameters}

    # Populate the dictionary using the open force field toolkit.
    for smiles in smiles_patterns:

        if smiles not in cached_smirks_parameters or parameter_tag not in cached_smirks_parameters[smiles]:

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
                parameter.smirks for parameter in force_field.label_molecules(topology)[0][parameter_tag].values()
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


class LogFilter(object):
    """

    """

    def __init__(self, data_set, message=None):

        self._initial_number_of_properties = -1
        self._data_set = data_set
        self._message = "were removed after filtering" if message is None else message

    def __enter__(self):
        self._initial_number_of_properties = self._data_set.number_of_properties

    def __exit__(self, exc_type, exc_value, exc_traceback):

        logger = logging.getLogger()

        logger.info(f"{self._initial_number_of_properties - self._data_set.number_of_properties} {self._message}")

        return True


log_filter = LogFilter
