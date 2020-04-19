import functools
import logging
import os
from multiprocessing import Pool

from nistdataselection.analysis.processing import processes_optimization_results
from nistdataselection.utils.utils import chemical_environment_codes

logger = logging.getLogger(__name__)


def main():

    n_processes = 4

    # Define the paths to the studies
    study_paths = {
        "rho_pure_h_vap": os.path.join(
            "..",
            "..",
            "pure_optimisation",
            "force_balance",
            "alcohol_ester",
            "rho_pure_h_vap",
        ),
        "h_mix_rho_x": os.path.join(
            "..",
            "..",
            "mixture_optimisation",
            "force_balance",
            "alcohol_ester",
            "h_mix_rho_x",
        ),
        "h_mix_rho_x_rho_pure": os.path.join(
            "..",
            "..",
            "pure_mixture_optimisation",
            "force_balance",
            "alcohol_ester",
            "h_mix_rho_x_rho_pure",
        ),
        "h_mix_rho_x_rho_pure_h_vap": os.path.join(
            "..",
            "..",
            "pure_mixture_optimisation",
            "force_balance",
            "alcohol_ester",
            "h_mix_rho_x_rho_pure_h_vap",
        ),
        "h_mix_v_excess": os.path.join(
            "..",
            "..",
            "mixture_optimisation",
            "force_balance",
            "alcohol_ester",
            "h_mix_v_excess",
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
