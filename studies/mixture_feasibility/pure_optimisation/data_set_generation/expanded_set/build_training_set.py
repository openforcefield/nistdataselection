import functools
import os
from collections import defaultdict

import pandas
from evaluator import unit
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.substances import Component

from nistdataselection.curation.filtering import filter_by_smiles
from nistdataselection.curation.selection import StatePoint
from nistdataselection.reporting import generate_report
from nistdataselection.utils import data_set_from_data_frame
from nistdataselection.utils.utils import data_frame_to_pdf


def main():

    training_set_smiles = [
        # Ethers
        "C1COCCO1",
        "C1CCOCC1",
        "COC(C)(C)C",
        "CC(C)OC(C)C",
        "CCCCOCCCC",
        # Ketones
        "O=C1CCCC1",
        "CCCC(C)=O",
        "O=C1CCCCC1",
        "O=C1CCCCCC1",
        # Alcohols
        "CO",
        "CCO",
        "CCCO",
        "CCCCO",
        "CC(C)(C)O",
        "CC(C)O",
        "CC(C)CO",
        # Esters
        "CC(=O)O",
        "COC=O",
        "CCOC(C)=O",
        "CCOC(=O)CC(=O)OCC",
        "CCCCOC(C)=O",
        "CCCOC(C)=O",
        # Alkanes
        "C1CCCCC1",
        "CCCCCC",
        "CC1CCCCC1",
        "CCCCCCC",
        "CC(C)CC(C)(C)C",
        "CCCCCCCCCC",
    ]

    # Ensure the smiles patterns are standardized.
    smiles = [Component(x).smiles for x in training_set_smiles]

    # Load in the Hvap data
    h_vap_data_frame = pandas.read_csv(
        os.path.join(
            "..",
            "..",
            "..",
            "data_availability",
            "sourced_h_vap_data",
            "enthalpy_of_vaporization_pure.csv",
        )
    )
    h_vap_data_frame = filter_by_smiles(
        h_vap_data_frame, smiles_to_include=smiles, smiles_to_exclude=None
    )

    h_vap_data_set = data_set_from_data_frame(h_vap_data_frame)

    # Load in the density data
    density_data_frame = pandas.read_csv(
        os.path.join(
            "..", "..", "..", "..", "shared", "filtered_data", "density_pure.csv",
        )
    )
    density_data_frame = filter_by_smiles(
        density_data_frame, smiles_to_include=smiles, smiles_to_exclude=None
    )

    density_data_set = data_set_from_data_frame(density_data_frame)

    # Retain the density measurements which were made closest to 298.15K and 1 atm.
    target_state_point = StatePoint(
        temperature=298.15 * unit.kelvin,
        pressure=1.0 * unit.atmosphere,
        mole_fractions=(1.0,),
    )

    final_data_set = PhysicalPropertyDataSet()

    for substance in density_data_set.substances:

        properties_per_state = defaultdict(list)

        # Refactor the properties into more convenient data structures.
        for physical_property in density_data_set.properties_by_substance(substance):

            state_point = StatePoint.from_physical_property(physical_property)
            properties_per_state[state_point].append(physical_property)

        # Sort the state points based on their distance to the target state.
        sorted_states_points = list(
            sorted(
                properties_per_state.keys(),
                key=functools.partial(
                    StatePoint.individual_distances, target_state_point
                ),
            )
        )

        final_data_set.add_properties(properties_per_state[sorted_states_points[0]][0])

    final_data_set.merge(h_vap_data_set)

    final_data_set.json("training_set.json", format=True)

    final_data_frame = final_data_set.to_pandas()
    final_data_frame.to_csv("training_set.csv", index=False)

    data_frame_to_pdf(final_data_frame, "training_set.pdf")

    generate_report("training_set.json")


if __name__ == "__main__":
    main()
