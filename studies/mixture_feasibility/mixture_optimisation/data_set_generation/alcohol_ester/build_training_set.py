import logging
import os

from evaluator import unit
from evaluator.properties import Density, EnthalpyOfMixing, ExcessMolarVolume

from nistdataselection.curation.filtering import filter_by_temperature
from nistdataselection.curation.selection import StatePoint, select_data_points
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import data_frame_to_pdf

logger = logging.getLogger(__name__)


def filter_common_data(output_directory):
    """Filter the common data to a smaller temperature range - this
    seems to help the state selection method get closer to the target
    states.
    """
    os.makedirs(os.path.join(output_directory, "h_mix_and_v_excess"), exist_ok=True)
    os.makedirs(
        os.path.join(output_directory, "h_mix_and_binary_density"), exist_ok=True
    )

    for property_type, substance_type in [
        (EnthalpyOfMixing, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
    ]:

        data_frame = load_processed_data_set(
            os.path.join(
                "..",
                "..",
                "data_availability",
                "data_by_environments",
                "alcohol_ester",
                "common_data",
                "h_mix_v_excess",
            ),
            property_type,
            substance_type,
        )
        data_frame = filter_by_temperature(
            data_frame, 290.0 * unit.kelvin, 305 * unit.kelvin
        )
        save_processed_data_set(
            os.path.join(output_directory, "h_mix_and_v_excess"),
            data_frame,
            property_type,
            substance_type,
        )

    for property_type, substance_type in [
        (EnthalpyOfMixing, SubstanceType.Binary),
        (Density, SubstanceType.Binary),
    ]:

        data_frame = load_processed_data_set(
            os.path.join(
                "..",
                "..",
                "data_availability",
                "data_by_environments",
                "alcohol_ester",
                "common_data",
                "h_mix_rho_x",
            ),
            property_type,
            substance_type,
        )
        data_frame = filter_by_temperature(
            data_frame, 290.0 * unit.kelvin, 305 * unit.kelvin
        )
        save_processed_data_set(
            os.path.join(output_directory, "h_mix_and_binary_density"),
            data_frame,
            property_type,
            substance_type,
        )


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

    filtered_directory = "filtered_common_data"
    os.makedirs(filtered_directory, exist_ok=True)

    filter_common_data(filtered_directory)

    output_directory = "training_sets"
    os.makedirs(output_directory, exist_ok=True)

    h_mix_v_excess = select_data_points(
        data_directory=os.path.join(filtered_directory, "h_mix_and_v_excess"),
        chosen_substances=substances,
        target_state_points={
            (EnthalpyOfMixing, SubstanceType.Binary): target_states,
            (ExcessMolarVolume, SubstanceType.Binary): target_states,
        },
    )

    h_mix_v_excess.json(
        os.path.join(output_directory, "h_mix_v_excess_training_set.json")
    )
    h_mix_v_excess = h_mix_v_excess.to_pandas()

    h_mix_v_excess.to_csv(
        os.path.join(output_directory, "h_mix_v_excess_training_set.csv"), index=False
    )
    data_frame_to_pdf(
        h_mix_v_excess,
        os.path.join(output_directory, "h_mix_v_excess_training_set.pdf"),
    )

    h_mix_density = select_data_points(
        data_directory=os.path.join(filtered_directory, "h_mix_and_binary_density"),
        chosen_substances=substances,
        target_state_points={
            (EnthalpyOfMixing, SubstanceType.Binary): target_states,
            (Density, SubstanceType.Binary): target_states,
        },
    )

    h_mix_density.json(
        os.path.join(output_directory, "h_mix_density_training_set.json")
    )
    h_mix_density = h_mix_density.to_pandas()

    h_mix_density.to_csv(
        os.path.join(output_directory, "h_mix_density_training_set.csv"), index=False
    )
    data_frame_to_pdf(
        h_mix_density, os.path.join(output_directory, "h_mix_density_training_set.pdf")
    )


if __name__ == "__main__":
    main()
