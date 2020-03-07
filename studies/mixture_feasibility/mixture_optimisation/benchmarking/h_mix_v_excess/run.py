from evaluator import unit
from evaluator.backends import QueueWorkerResources
from evaluator.backends.dask import DaskLSFBackend
from evaluator.client import ConnectionOptions, EvaluatorClient
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.forcefield import SmirnoffForceFieldSource
from evaluator.server import EvaluatorServer
from evaluator.utils import setup_timestamp_logging


def main():

    setup_timestamp_logging()

    # Load in the force field
    force_field_path = "openff-1.0.0-refit.offxml"
    force_field_source = SmirnoffForceFieldSource.from_path(force_field_path)

    # Load in the test set.
    data_set = PhysicalPropertyDataSet.from_json("full_set.json")

    # Set up a server object to run the calculations using.
    working_directory = "working_directory"

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
            port=8002,
        )

        with server:

            # Request the estimates.
            client = EvaluatorClient(ConnectionOptions(server_port=8002))

            request, _ = client.request_estimate(
                property_set=data_set, force_field_source=force_field_source,
            )

            # Wait for the results.
            results, _ = request.results(True, 5)
            results.json(f"results.json")


if __name__ == "__main__":
    main()
