import logging
import os

from evaluator import unit
from evaluator.properties import Density, EnthalpyOfMixing, ExcessMolarVolume

from nistdataselection.curation.selection import StatePoint, select_data_points
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import data_frame_to_pdf

logger = logging.getLogger(__name__)


def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    substances = [
        # Methanol
        ("CO", "COC=O"),  # Methyl formate
        ("CO", "CCOC(=O)CC(=O)OCC"),  # Diethyl Malonate
        # Ethanol
        ("CCO", "CC(=O)O"),  # Acetic acid
        ("CCO", "CCOC(C)=O"),  # Ethyl acetate
        ("CCO", "CCOC(=O)CC(=O)OCC"),  # Diethyl Malonate
        # Butanol
        ("CCCCO", "CCOC(=O)CC(=O)OCC"),  # Diethyl Malonate
        # Isopropanol
        ("CC(C)O", "CCOC(=O)CC(=O)OCC"),  # Diethyl Malonate
        # Isobutanol
        ("CC(C)CO", "CCOC(=O)CC(=O)OCC"),  # Diethyl Malonate
        # Tert-butanol
        ("CC(C)(C)O", "COC=O"),  # Methyl formate
        ("CC(C)(C)O", "CCCCOC(C)=O"),  # Butyl acetate
    ]

    substances = [tuple(sorted(x)) for x in substances]

    target_states = [
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.25, 0.75)),
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.50, 0.50)),
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.75, 0.25)),
    ]

    h_mix_v_excess = select_data_points(
        data_directory=os.path.join("common_data", "h_mix_and_v_excess"),
        chosen_substances=substances,
        target_state_points={
            (EnthalpyOfMixing, SubstanceType.Binary): target_states,
            (ExcessMolarVolume, SubstanceType.Binary): target_states,
        },
    )

    h_mix_v_excess.json("h_mix_v_excess_training_set.json")
    h_mix_v_excess = h_mix_v_excess.to_pandas()

    h_mix_v_excess.to_csv("h_mix_v_excess_training_set.csv", index=False)
    data_frame_to_pdf(h_mix_v_excess, "h_mix_v_excess_training_set.pdf")

    h_mix_density = select_data_points(
        data_directory=os.path.join("common_data", "h_mix_and_binary_density"),
        chosen_substances=substances,
        target_state_points={
            (EnthalpyOfMixing, SubstanceType.Binary): target_states,
            (Density, SubstanceType.Binary): target_states,
        },
    )

    h_mix_density.json("h_mix_density_training_set.json")
    h_mix_density = h_mix_density.to_pandas()

    h_mix_density.to_csv("h_mix_density_training_set.csv", index=False)
    data_frame_to_pdf(h_mix_density, "h_mix_density_training_set.pdf")


if __name__ == "__main__":
    main()
