#!/usr/bin/env python3
import shutil
from os import path

from evaluator import unit
from evaluator.backends import QueueWorkerResources
from evaluator.backends.dask import DaskLSFBackend
from evaluator.server import EvaluatorServer


def main():

    working_directory = "working_directory"

    # Remove any existing data.
    if path.isdir(working_directory):
        shutil.rmtree(working_directory)

    # Set up a backend to run the calculations on. This assume running
    # on a HPC resources with the LSF queue system installed.
    queue_resources = QueueWorkerResources(
        number_of_threads=1,
        number_of_gpus=1,
        preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
        per_thread_memory_limit=5 * unit.gigabyte,
        wallclock_time_limit="05:59",
    )

    worker_script_commands = ["conda activate forcebalance", "module load cuda/10.1"]

    calculation_backend = DaskLSFBackend(
        minimum_number_of_workers=1,
        maximum_number_of_workers=50,
        resources_per_worker=queue_resources,
        queue_name="gpuqueue",
        setup_script_commands=worker_script_commands,
        adaptive_interval="1000ms",
    )

    with calculation_backend:

        server = EvaluatorServer(
            calculation_backend=calculation_backend,
            working_directory=working_directory,
            port=8005,
        )

        # Tell the server to start listening for estimation requests.
        server.start()


if __name__ == "__main__":
    main()
