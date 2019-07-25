"""Utilities to help with processing the NIST data sets.
"""
import shutil
import subprocess
import tempfile
from enum import Enum

from propertyestimator.backends import DaskLocalClusterBackend, QueueWorkerResources, DaskLSFBackend
from simtk import unit


class BackendType(Enum):
    Local = 'Local'
    LSF = 'LSF'


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
        calculation_backend = DaskLocalClusterBackend(number_of_workers=number_of_workers)

    elif backend_type == BackendType.LSF:

        queue_resources = QueueWorkerResources(number_of_threads=1,
                                               per_thread_memory_limit=12 * (unit.giga * unit.byte),
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
    oe_molecule.SetTitle(off_molecule.to_smiles())

    oedepict.OEPrepareDepiction(oe_molecule)

    options = oedepict.OE2DMolDisplayOptions(256, 256, oedepict.OEScale_AutoScale)

    display = oedepict.OE2DMolDisplay(oe_molecule, options)
    oedepict.OERenderMolecule(file_path, display)
