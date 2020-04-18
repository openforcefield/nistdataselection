import functools
import logging
import os
from multiprocessing import Pool

from nistdataselection.analysis.processing import processes_benchmark_results
from nistdataselection.utils.utils import chemical_environment_codes

logger = logging.getLogger(__name__)


def main():

    n_processes = 4

    # Define the paths to the studies
    study_paths = {
        "rho_pure_h_vap": os.path.join(
            "..", "..", "results", "expanded_set", "rho_pure_h_vap.json"
        ),
        "h_mix_rho_x": os.path.join(
            "..", "..", "results", "expanded_set", "h_mix_rho_x.json"
        ),
        "h_mix_rho_x_rho_pure": os.path.join(
            "..", "..", "results", "expanded_set", "h_mix_rho_x_rho_pure.json"
        ),
        "openff-1.0.0": os.path.join(
            "..", "..", "results", "expanded_set", "openff-1.0.0.json"
        ),
    }

    reference_set_path = os.path.join(
        "..", "..", "data_set_generation", "expanded_set", "test_sets", "full_set.json"
    )

    environments = {
        "alcohol": [
            chemical_environment_codes["hydroxy"],
            chemical_environment_codes["alcohol"],
        ],
        "ester": [
            chemical_environment_codes["caboxylic_acid"],
            chemical_environment_codes["ester"],
        ],
        "ether": [chemical_environment_codes["ether"]],
        "ketone": [chemical_environment_codes["ketone"]],
        "alkane": [""],
    }

    with Pool(n_processes) as pool:

        list(
            pool.starmap(
                functools.partial(
                    processes_benchmark_results,
                    reference_path=reference_set_path,
                    environments=environments,
                    root_output_directory="",
                ),
                study_paths.items(),
            )
        )


if __name__ == "__main__":
    main()
