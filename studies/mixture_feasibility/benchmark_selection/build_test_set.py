"""A script to filter out all of the training set
compounds from the available data.
"""
import functools
import os
from collections import defaultdict

import pandas
from evaluator import unit
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.properties import Density, EnthalpyOfMixing, ExcessMolarVolume

from nistdataselection.curation.filtering import (
    filter_by_smiles,
    filter_by_substance_composition,
    filter_by_temperature,
)
from nistdataselection.curation.selection import StatePoint, select_data_points
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils import SubstanceType, data_set_from_data_frame
from nistdataselection.utils.utils import data_frame_to_pdf


def build_pure_set(pure_training_set_smiles):

    # Load in the Hvap data
    h_vap_data_frame = pandas.read_csv(
        os.path.join(
            "..", "data_availability", "sourced_h_vap_data", "alcohol_ester_h_vap.csv"
        )
    )
    h_vap_data_frame = filter_by_smiles(
        h_vap_data_frame,
        smiles_to_include=None,
        smiles_to_exclude=pure_training_set_smiles,
    )

    h_vap_data_set = data_set_from_data_frame(h_vap_data_frame)

    # Pull out the chosen smiles patterns
    chosen_smiles = [*h_vap_data_frame["Component 1"]]

    # # Load in the density data
    density_data_frame = pandas.read_csv(
        os.path.join(
            "..", "data_availability", "all_alcohol_ester_data", "density_pure.csv"
        )
    )
    density_data_frame = filter_by_smiles(
        density_data_frame, smiles_to_include=chosen_smiles, smiles_to_exclude=None
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

    return final_data_set


def filter_common_data(mixture_training_set_smiles, output_directory):
    """Filter the common data to a smaller temperature range - this
    seems to help the state selection method get closer to the target
    states.
    """
    os.makedirs(os.path.join(output_directory), exist_ok=True)

    for property_type, substance_type in [
        (EnthalpyOfMixing, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
        (Density, SubstanceType.Binary),
    ]:

        folder_name = (
            "h_mix_and_v_excess"
            if property_type != Density
            else "h_mix_and_binary_density"
        )

        data_frame = load_processed_data_set(
            os.path.join("..", "data_availability", "common_data", folder_name),
            property_type,
            substance_type,
        )
        data_frame = filter_by_temperature(
            data_frame, 290.0 * unit.kelvin, 305 * unit.kelvin
        )
        data_frame = filter_by_substance_composition(
            data_frame, None, mixture_training_set_smiles
        )

        save_processed_data_set(
            output_directory, data_frame, property_type, substance_type,
        )


def build_mixture_set(mixture_training_set_smiles):

    target_states = [
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.25, 0.75)),
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.50, 0.50)),
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.75, 0.25)),
    ]

    filtered_directory = "filtered_mixture_data"
    filter_common_data(mixture_training_set_smiles, filtered_directory)

    output_directory = "training_sets"
    os.makedirs(output_directory, exist_ok=True)

    mixture_set = select_data_points(
        data_directory=filtered_directory,
        chosen_substances=None,
        target_state_points={
            (EnthalpyOfMixing, SubstanceType.Binary): target_states,
            (ExcessMolarVolume, SubstanceType.Binary): target_states,
            (Density, SubstanceType.Binary): target_states,
        },
    )

    return mixture_set


def main():

    output_directory = "selected_sets"
    os.makedirs(output_directory, exist_ok=True)

    # Determine which compounds were used during training.
    h_mix_v_excess_set = PhysicalPropertyDataSet.from_json(
        os.path.join(
            "..",
            "pure_mixture_optimisation",
            "force_balance",
            "h_mix_v_excess_rho_pure_h_vap",
            "targets",
            "mixture_data",
            "training_set.json",
        )
    )
    h_mix_rho_x_set = PhysicalPropertyDataSet.from_json(
        os.path.join(
            "..",
            "pure_mixture_optimisation",
            "force_balance",
            "h_mix_rho_x_rho_pure_h_vap",
            "targets",
            "mixture_data",
            "training_set.json",
        )
    )

    all_substance_smiles = [
        *((x.smiles for x in y) for y in h_mix_v_excess_set.substances),
        *((x.smiles for x in y) for y in h_mix_rho_x_set.substances),
    ]
    all_substance_smiles = [tuple(sorted(x)) for x in all_substance_smiles]

    unique_substance_smiles = set(all_substance_smiles)

    pure_substance_smiles = [x for x in unique_substance_smiles if len(x) == 1]
    binary_substance_smiles = [x for x in unique_substance_smiles if len(x) == 2]

    # Select a pure test set
    pure_test_set = build_pure_set(pure_substance_smiles)
    pure_test_set.json(os.path.join(output_directory, "pure_set.json"))

    pure_test_pandas = pure_test_set.to_pandas()

    pure_test_pandas.to_csv(os.path.join(output_directory, "pure_set.csv"), index=False)
    data_frame_to_pdf(
        pure_test_pandas, os.path.join(output_directory, "pure_set.pdf"),
    )

    # Select a mixture test set
    mixture_test_set = build_mixture_set(binary_substance_smiles)
    mixture_test_set.json(os.path.join(output_directory, "mixture_set.json"))

    mixture_test_pandas = mixture_test_set.to_pandas()

    mixture_test_pandas.to_csv(
        os.path.join(output_directory, "mixture_set.csv"), index=False
    )
    data_frame_to_pdf(
        mixture_test_pandas, os.path.join(output_directory, "mixture_set.pdf"),
    )

    # Combine the two sets
    full_data_set = PhysicalPropertyDataSet()
    full_data_set.merge(pure_test_set)
    full_data_set.merge(mixture_test_set)

    full_data_set.json(os.path.join(output_directory, "full_set.json"))

    full_set_pandas = full_data_set.to_pandas()

    full_set_pandas.to_csv(os.path.join(output_directory, "full_set.csv"), index=False)


if __name__ == "__main__":
    main()
