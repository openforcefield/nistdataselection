import functools
import logging
import os
from multiprocessing import Pool

from nistdataselection.analysis.processing import processes_optimization_results
from nistdataselection.utils.utils import chemical_environment_codes

logger = logging.getLogger(__name__)


def main():

    n_processes = 2

    # Define the paths to the studies
    study_paths = {
        "rho_pure_h_vap": os.path.join(
            "..",
            "..",
            "pure_optimisation",
            "force_balance",
            "expanded_set",
            "rho_pure_h_vap",
        ),
        "h_mix_rho_x": os.path.join(
            "..",
            "..",
            "mixture_optimisation",
            "force_balance",
            "expanded_set",
            "h_mix_rho_x",
        ),
        "h_mix_rho_x_rho_pure": os.path.join(
            "..",
            "..",
            "pure_mixture_optimisation",
            "force_balance",
            "expanded_set",
            "h_mix_rho_x_rho_pure",
        ),
        "h_mix_rho_x_rho_pure_h_vap": os.path.join(
            "..",
            "..",
            "pure_mixture_optimisation",
            "force_balance",
            "expanded_set",
            "h_mix_rho_x_rho_pure_h_vap",
        ),
    }

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
                    processes_optimization_results,
                    environments=environments,
                    root_output_directory="",
                ),
                study_paths.items(),
            )
        )


if __name__ == "__main__":
    main()
