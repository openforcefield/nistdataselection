import logging
import os

import pandas
from evaluator import unit
from evaluator.properties import Density, EnthalpyOfMixing

from nistdataselection.curation.filtering import filter_by_substance_composition
from nistdataselection.curation.selection import StatePoint, select_data_points
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.reporting import generate_report
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import (
    data_frame_to_pdf,
    property_to_file_name,
    smiles_to_pdf,
)

logger = logging.getLogger(__name__)


def filter_common_data(output_directory, substances):
    """Filter the common data to a smaller temperature range - this
    seems to help the state selection method get closer to the target
    states.
    """
    os.makedirs(os.path.join(output_directory, "h_mix_and_rho_x"), exist_ok=True)

    for property_type, substance_type in [
        (EnthalpyOfMixing, SubstanceType.Binary),
        (Density, SubstanceType.Binary),
    ]:

        data_frames = []

        for environment_mix in [
            "alcohol_ester",
            "alcohol_alkane",
            "ether_alkane",
            "ether_ketone",
        ]:

            data_frame = load_processed_data_set(
                os.path.join(
                    "..",
                    "..",
                    "..",
                    "data_availability",
                    "data_by_environments",
                    environment_mix,
                    "common_data",
                    "h_mix_rho_x",
                ),
                property_type,
                substance_type,
            )

            data_frame = filter_by_substance_composition(data_frame, substances, None)

            data_frame = data_frame[
                (data_frame["Mole Fraction 1"] > 0.10)
                & (data_frame["Mole Fraction 1"] < 0.90)
            ]

            data_frames.append(data_frame)

        full_data_frame = pandas.concat(data_frames)

        save_processed_data_set(
            os.path.join(output_directory, "h_mix_and_rho_x"),
            full_data_frame,
            property_type,
            substance_type,
        )
        data_frame_to_pdf(
            full_data_frame,
            os.path.join(
                output_directory,
                "h_mix_and_rho_x",
                property_to_file_name(property_type, substance_type) + ".pdf",
            ),
        )


def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    smiles_map = {
        # Ethers
        "1,4-dioxane": "C1COCCO1",
        "oxane": "C1CCOCC1",
        "methyl tert butyl ether": "COC(C)(C)C",
        "diisopropyl ether": "CC(C)OC(C)C",
        "dibuytl ether": "CCCCOCCCC",
        # Ketones
        "cyclopentanone": "O=C1CCCC1",
        "2-pentanone": "CCCC(C)=O",
        "cyclohexanone": "O=C1CCCCC1",
        "cycloheptanone": "O=C1CCCCCC1",
        # Alcohols
        "methanol": "CO",
        "ethanol": "CCO",
        "propanol": "CCCO",
        "butanol": "CCCCO",
        "propan-2-ol": "CC(C)O",
        "2-Methylpropan-1-ol": "CC(C)CO",
        "2-Methylpropan-2-ol": "CC(C)(C)O",
        # Esters / acids
        "acetic acid": "CC(=O)O",
        "methyl formate": "COC=O",
        "ethyl acetate": "CCOC(C)=O",
        "propyl acetate": "CCCOC(C)=O",
        "butyl acetate": "CCCCOC(C)=O",
        "diethyl propanedioate": "CCOC(=O)CC(=O)OCC",
        # Alkanes
        "cyclohexane": "C1CCCCC1",
        "hexane": "CCCCCC",
        "methylcyclohexane": "CC1CCCCC1",
        "heptane": "CCCCCCC",
        "iso-octane": "CC(C)CC(C)(C)C",
        "decane": "CCCCCCCCCC",
    }

    substances = [
        # Ether - Alkane
        (smiles_map["dibuytl ether"], smiles_map["iso-octane"]),
        (smiles_map["oxane"], smiles_map["heptane"]),
        (smiles_map["methyl tert butyl ether"], smiles_map["decane"]),
        (smiles_map["diisopropyl ether"], smiles_map["iso-octane"]),
        (smiles_map["diisopropyl ether"], smiles_map["heptane"]),
        (smiles_map["oxane"], smiles_map["hexane"]),
        (smiles_map["oxane"], smiles_map["cyclohexane"]),
        # Alcohol - Alkane
        (smiles_map["propanol"], smiles_map["cyclohexane"]),
        (smiles_map["propanol"], smiles_map["iso-octane"]),
        (smiles_map["propanol"], smiles_map["methylcyclohexane"]),
        (smiles_map["butanol"], smiles_map["iso-octane"]),
        (smiles_map["butanol"], smiles_map["hexane"]),
        (smiles_map["butanol"], smiles_map["methylcyclohexane"]),
        (smiles_map["butanol"], smiles_map["heptane"]),
        (smiles_map["ethanol"], smiles_map["iso-octane"]),
        (smiles_map["ethanol"], smiles_map["heptane"]),
        # Ether - Ketone
        (smiles_map["oxane"], smiles_map["cyclopentanone"]),
        (smiles_map["oxane"], smiles_map["cyclohexanone"]),
        (smiles_map["oxane"], smiles_map["2-pentanone"]),
        (smiles_map["1,4-dioxane"], smiles_map["cyclopentanone"]),
        (smiles_map["1,4-dioxane"], smiles_map["cyclohexanone"]),
        (smiles_map["1,4-dioxane"], smiles_map["2-pentanone"]),
        (smiles_map["1,4-dioxane"], smiles_map["cycloheptanone"]),
        # Alcohol - Ester ( / acid)
        (smiles_map["methanol"], smiles_map["methyl formate"]),
        (smiles_map["methanol"], smiles_map["diethyl propanedioate"]),
        (smiles_map["ethanol"], smiles_map["acetic acid"]),
        (smiles_map["ethanol"], smiles_map["ethyl acetate"]),
        (smiles_map["ethanol"], smiles_map["diethyl propanedioate"]),
        (smiles_map["butanol"], smiles_map["diethyl propanedioate"]),
        (smiles_map["propan-2-ol"], smiles_map["diethyl propanedioate"]),
        (smiles_map["2-Methylpropan-1-ol"], smiles_map["diethyl propanedioate"]),
        (smiles_map["2-Methylpropan-2-ol"], smiles_map["methyl formate"]),
        (smiles_map["2-Methylpropan-2-ol"], smiles_map["butyl acetate"]),
    ]

    substances = [tuple(sorted(x)) for x in substances]

    smiles_to_pdf(substances, "all_substances.pdf")

    target_states = [
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.25, 0.75)),
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.50, 0.50)),
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.75, 0.25)),
    ]

    filtered_directory = "filtered_common_data"
    os.makedirs(filtered_directory, exist_ok=True)

    filter_common_data(filtered_directory, substances)

    output_directory = "training_sets"
    os.makedirs(output_directory, exist_ok=True)

    h_mix_rho_x = select_data_points(
        data_directory=os.path.join(filtered_directory, "h_mix_and_rho_x"),
        chosen_substances=None,
        target_state_points={
            (EnthalpyOfMixing, SubstanceType.Binary): target_states,
            (Density, SubstanceType.Binary): target_states,
        },
    )

    h_mix_rho_x.json(os.path.join(output_directory, "h_mix_rho_x_training_set.json"))
    h_mix_rho_x = h_mix_rho_x.to_pandas()

    h_mix_rho_x.to_csv(
        os.path.join(output_directory, "h_mix_rho_x_training_set.csv"), index=False
    )
    data_frame_to_pdf(
        h_mix_rho_x, os.path.join(output_directory, "h_mix_rho_x_training_set.pdf"),
    )

    generate_report(os.path.join(output_directory, "h_mix_rho_x_training_set.json"))


if __name__ == "__main__":
    main()
