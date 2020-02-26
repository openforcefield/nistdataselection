import logging
import os

from evaluator.datasets import PhysicalPropertyDataSet

logger = logging.getLogger(__name__)


def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    output_directory = "training_sets"
    os.makedirs(output_directory, exist_ok=True)

    pure_density_h_vap = PhysicalPropertyDataSet.from_json(
        "../../pure_optimisation/data_set_generation/training_set.json"
    )

    pure_density = PhysicalPropertyDataSet.from_json(
        "../../pure_optimisation/data_set_generation/training_set.json"
    )
    pure_density.filter_by_property_types("Density")

    h_mix_v_excess = PhysicalPropertyDataSet.from_json(
        "../../mixture_optimisation/data_set_generation/"
        "training_sets/h_mix_v_excess_training_set.json"
    )
    h_mix_binary_density = PhysicalPropertyDataSet.from_json(
        "../../mixture_optimisation/data_set_generation/"
        "training_sets/h_mix_density_training_set.json"
    )

    h_mix_binary_density_pure_density = PhysicalPropertyDataSet()
    h_mix_binary_density_pure_density.merge(pure_density)
    h_mix_binary_density_pure_density.merge(h_mix_binary_density)
    h_mix_binary_density_pure_density.json(
        os.path.join(output_directory, "h_mix_binary_density_pure_density.json")
    )
    h_mix_binary_density_pure_density.to_pandas().to_csv(
        os.path.join(output_directory, "h_mix_binary_density_pure_density.csv")
    )

    h_mix_v_excess_pure_density = PhysicalPropertyDataSet()
    h_mix_v_excess_pure_density.merge(pure_density)
    h_mix_v_excess_pure_density.merge(h_mix_v_excess)
    h_mix_v_excess_pure_density.json(
        os.path.join(output_directory, "h_mix_v_excess_pure_density.json")
    )
    h_mix_v_excess_pure_density.to_pandas().to_csv(
        os.path.join(output_directory, "h_mix_v_excess_pure_density.csv")
    )
    
    h_mix_binary_density_pure_density_h_vap = PhysicalPropertyDataSet()
    h_mix_binary_density_pure_density_h_vap.merge(pure_density_h_vap)
    h_mix_binary_density_pure_density_h_vap.merge(h_mix_binary_density)
    h_mix_binary_density_pure_density_h_vap.json(
        os.path.join(output_directory, "h_mix_binary_density_pure_density_h_vap.json")
    )
    h_mix_binary_density_pure_density_h_vap.to_pandas().to_csv(
        os.path.join(output_directory, "h_mix_binary_density_pure_density_h_vap.csv")
    )
    
    h_mix_v_excess_pure_density_h_vap = PhysicalPropertyDataSet()
    h_mix_v_excess_pure_density_h_vap.merge(pure_density_h_vap)
    h_mix_v_excess_pure_density_h_vap.merge(h_mix_v_excess)
    h_mix_v_excess_pure_density_h_vap.json(
        os.path.join(output_directory, "h_mix_v_excess_pure_density_h_vap.json")
    )
    h_mix_v_excess_pure_density_h_vap.to_pandas().to_csv(
        os.path.join(output_directory, "h_mix_v_excess_pure_density_h_vap.csv")
    )


if __name__ == "__main__":
    main()
