"""Utilities to help with processing the NIST data sets.
"""
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
