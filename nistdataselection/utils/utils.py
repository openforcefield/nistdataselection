"""Utilities to help with processing the NIST data sets.
"""
import shutil
import subprocess
import tempfile
from collections import defaultdict
from enum import Enum

from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils import UndefinedStereochemistryError
from propertyestimator import unit
from propertyestimator.backends import DaskLocalCluster, QueueWorkerResources, DaskLSFBackend


class BackendType(Enum):
    Local = 'Local'
    LSF = 'LSF'


class SubstanceType(Enum):
    """An enum which encodes the names used for substances
    with different numbers of components.
    """
    Pure = 'pure'
    Binary = 'binary'
    Ternary = 'ternary'


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


def setup_parallel_backend(backend_type=BackendType.Local,
                           number_of_workers=1,
                           lsf_queue='cpuqueue',
                           lsf_worker_commands=None):
    """Sets up the `PropertyEstimatorBackend` that will be used to distribute
    the data extraction from the xml files over multiple threads / compute nodes.

    Parameters
    ----------
    backend_type: BackendType
        The type of backend to set up.
    number_of_workers: int
        The number of workers to distribute the data extraction
        over. If the `backend_type` is set to `BackendType.Local`,
        this should be set to the number of CPU's available on your
        machine. Otherwise, this represents the number of compute
        workers which will be spun up in your LSF queueing system.
    lsf_queue: str, optional
        The queue to create the compute workers in when `backend_type`
        is set to `BackendType.LSF`.
    lsf_worker_commands: list of str
        A list of commands to run on each spun up worker (such as setting
        up the correct conda environment) when `backend_type` is set to
        `BackendType.LSF`.

    Returns
    -------
    PropertyEstimatorBackend
        The created and started backend.
    """

    calculation_backend = None

    if backend_type == BackendType.Local:
        calculation_backend = DaskLocalCluster(number_of_workers=number_of_workers)

    elif backend_type == BackendType.LSF:

        queue_resources = QueueWorkerResources(number_of_threads=1,
                                               per_thread_memory_limit=12 * unit.gigabyte,
                                               wallclock_time_limit="01:30")

        if lsf_worker_commands is None:
            lsf_worker_commands = []

        calculation_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                             maximum_number_of_workers=number_of_workers,
                                             resources_per_worker=queue_resources,
                                             queue_name=lsf_queue,
                                             setup_script_commands=lsf_worker_commands,
                                             adaptive_interval='1000ms')

    calculation_backend.start()

    return calculation_backend


def analyse_functional_groups(smiles):
    """Employs checkmol to determine which chemical moieties
    are encoded by a given smiles pattern.

    Parameters
    ----------
    smiles: str
        The smiles pattern to examine.

    Returns
    -------
    list of str, optional
        A list of matching chemical moiety descriptors. If
        checkmol did not execute correctly, returns None.
    """
    from openforcefield.topology import Molecule

    # Make sure the checkmol utility has been installed separately.
    if shutil.which('checkmol') is None:

        raise FileNotFoundError('checkmol was not found on this machine. Visit '
                                'http://merian.pch.univie.ac.at/~nhaider/cheminf/cmmm.html '
                                'to obtain it.')

    molecule = Molecule.from_smiles(smiles)

    # Save the smile pattern out as an SDF file, ready
    # to use as input to checkmol.
    with tempfile.NamedTemporaryFile() as file:

        molecule.to_file(file, 'SDF')

        # Execute checkmol.
        result = subprocess.check_output(['checkmol', file.name],
                                         stderr=subprocess.STDOUT).decode()

    groups = None

    # Turn the string output into a list of moieties.
    if len(result) > 0:
        groups = list(filter(None, result.replace('\n', '').split(';')))

    return groups


def smiles_to_png(smiles, file_path):
    """Creates a png image of the 2D representation of
    a given smiles pattern.

    Parameters
    ----------
    smiles: str
        The smiles pattern to generate the png of.
    file_path: str
        The path of the output png file.
    """

    from openeye import oedepict
    from openforcefield.topology import Molecule

    off_molecule = Molecule.from_smiles(smiles)
    oe_molecule = off_molecule.to_openeye()
    # oe_molecule.SetTitle(off_molecule.to_smiles())

    oedepict.OEPrepareDepiction(oe_molecule)

    options = oedepict.OE2DMolDisplayOptions(200, 200, oedepict.OEScale_AutoScale)

    display = oedepict.OE2DMolDisplay(oe_molecule, options)
    oedepict.OERenderMolecule(file_path, display)


cached_smirks_parameters = {}


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

    force_field = ForceField('smirnoff99Frosst-1.0.9.offxml')
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

    inverted_dictionary = defaultdict(list)

    for key in dictionary:
        for list_value in dictionary[key]:

            inverted_dictionary[list_value].append(key)

    return inverted_dictionary
