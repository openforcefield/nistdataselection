"""A script to filter out all of the training set
compounds from the available data.
"""
import functools
import os
from collections import defaultdict
from tempfile import TemporaryDirectory

import pandas
from evaluator import unit
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection.curation.filtering import filter_by_smiles
from nistdataselection.curation.selection import StatePoint, select_data_points
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.pandas import (
    data_frame_to_smiles_tuples,
    data_set_from_data_frame,
)
from nistdataselection.utils.utils import data_frame_to_pdf, smiles_to_pdf


def build_pure_set():

    # Load in the Hvap data
    h_vap_data_frame = load_processed_data_set(
        "filtered_data", EnthalpyOfVaporization, SubstanceType.Pure
    )

    # Filter out water
    h_vap_data_frame = filter_by_smiles(
        h_vap_data_frame, smiles_to_include=None, smiles_to_exclude=["O"]
    )

    h_vap_data_set = data_set_from_data_frame(h_vap_data_frame)

    density_data_frame = load_processed_data_set(
        "filtered_data", Density, SubstanceType.Pure
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


def build_mixture_set():

    # Pull out data for all of the mixtures whereby neither component was
    # in the training set.
    with TemporaryDirectory() as working_directory:

        for property_type, substance_type in [
            (EnthalpyOfMixing, SubstanceType.Binary),
            (ExcessMolarVolume, SubstanceType.Binary),
            (Density, SubstanceType.Binary),
        ]:

            pair_data_frames = []

            for pair_type in ["alcohol_only", "alcohol_ester", "ester_ester"]:

                data_directory = os.path.join(
                    f"partitioned_{pair_type}_data", "neither_in_training"
                )
                pair_data_frames.append(
                    load_processed_data_set(
                        data_directory, property_type, substance_type
                    )
                )

            pair_data_frame = pandas.concat(
                pair_data_frames, ignore_index=True, sort=False
            )

            save_processed_data_set(
                working_directory, pair_data_frame, property_type, substance_type
            )

        target_states = [
            StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.25, 0.75)),
            StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.50, 0.50)),
            StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.75, 0.25)),
        ]

        mixture_set = select_data_points(
            data_directory=working_directory,
            chosen_substances=None,
            target_state_points={
                (EnthalpyOfMixing, SubstanceType.Binary): target_states,
                (ExcessMolarVolume, SubstanceType.Binary): target_states,
                (Density, SubstanceType.Binary): target_states,
            },
        )

    return mixture_set


def main():

    output_directory = "test_sets"
    os.makedirs(output_directory, exist_ok=True)

    # Select a pure test set
    pure_test_set = build_pure_set()
    pure_test_set.json(os.path.join(output_directory, "pure_set.json"))

    pure_test_pandas = pure_test_set.to_pandas()

    pure_test_pandas.to_csv(os.path.join(output_directory, "pure_set.csv"), index=False)
    data_frame_to_pdf(
        pure_test_pandas, os.path.join(output_directory, "pure_set.pdf"),
    )

    # Select a mixture test set
    mixture_test_set = build_mixture_set()
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

    # Find the overlap of components of the pure and mixture sets.
    unique_pure_systems = data_frame_to_smiles_tuples(pure_test_pandas)
    unique_pure_components = set(x for y in unique_pure_systems for x in y)

    unique_mixture_systems = data_frame_to_smiles_tuples(mixture_test_pandas)
    unique_mixture_components = set(x for y in unique_mixture_systems for x in y)

    smiles_to_pdf(
        [*unique_mixture_components],
        os.path.join(output_directory, "mixture_set_only_components.pdf"),
    )

    common_components = unique_pure_components.intersection(unique_mixture_components)

    smiles_to_pdf(
        [*common_components],
        os.path.join(output_directory, "pure_and_mixture_components.pdf"),
    )


if __name__ == "__main__":
    main()
