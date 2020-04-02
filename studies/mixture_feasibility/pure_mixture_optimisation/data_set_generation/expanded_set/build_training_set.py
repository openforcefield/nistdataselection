import logging
import os

from evaluator.datasets import PhysicalPropertyDataSet

logger = logging.getLogger(__name__)


def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    output_directory = "training_sets"
    os.makedirs(output_directory, exist_ok=True)

    rho_pure_h_vap = PhysicalPropertyDataSet.from_json(
        "../../../pure_optimisation/data_set_generation/expanded_set/training_set.json"
    )

    rho_pure = PhysicalPropertyDataSet.from_json(
        "../../../pure_optimisation/data_set_generation/expanded_set/training_set.json"
    )
    rho_pure.filter_by_property_types("Density")

    h_mix_rho_x = PhysicalPropertyDataSet.from_json(
        "../../../mixture_optimisation/data_set_generation/"
        "expanded_set/training_sets/h_mix_rho_x_training_set.json"
    )

    h_mix_rho_x_rho_pure = PhysicalPropertyDataSet()
    h_mix_rho_x_rho_pure.merge(rho_pure)
    h_mix_rho_x_rho_pure.merge(h_mix_rho_x)
    h_mix_rho_x_rho_pure.json(
        os.path.join(output_directory, "h_mix_rho_x_rho_pure.json")
    )
    h_mix_rho_x_rho_pure.to_pandas().to_csv(
        os.path.join(output_directory, "h_mix_rho_x_rho_pure.csv")
    )

    h_mix_rho_x_rho_pure_h_vap = PhysicalPropertyDataSet()
    h_mix_rho_x_rho_pure_h_vap.merge(rho_pure_h_vap)
    h_mix_rho_x_rho_pure_h_vap.merge(h_mix_rho_x)
    h_mix_rho_x_rho_pure_h_vap.json(
        os.path.join(output_directory, "h_mix_rho_x_rho_pure_h_vap.json")
    )
    h_mix_rho_x_rho_pure_h_vap.to_pandas().to_csv(
        os.path.join(output_directory, "h_mix_rho_x_rho_pure_h_vap.csv")
    )


if __name__ == "__main__":
    main()
