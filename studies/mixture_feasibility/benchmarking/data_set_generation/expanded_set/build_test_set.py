"""
"""
import functools
import json
import logging
import os
from multiprocessing.pool import Pool

import pandas
from evaluator import unit
from evaluator.properties import Density, EnthalpyOfMixing, ExcessMolarVolume
from openeye.oegraphsim import OEFingerPrint, OEFPType_Tree, OEMakeFP, OETanimoto
from openforcefield.topology import Molecule

from nistdataselection.curation.filtering import (
    filter_by_smirks,
    filter_by_substance_composition,
    filter_by_temperature,
    filter_undefined_stereochemistry,
    filter_by_elements)
from nistdataselection.processing import load_processed_data_set
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import property_to_file_name, smiles_to_pdf

logger = logging.getLogger(__name__)


def load_training_mixtures():
    """
    """

    training_set = pandas.read_csv(
        os.path.join(
            "..",
            "..",
            "..",
            "mixture_optimisation",
            "data_set_generation",
            "expanded_set",
            "training_sets",
            "h_mix_rho_x_training_set.csv",
        )
    )

    mixtures = {
        tuple(sorted((x["Component 1"], x["Component 2"])))
        for _, x in training_set.iterrows()
    }

    return mixtures


def filter_data(data_frame):

    # Filter to be closer to ambient.
    data_frame = filter_by_temperature(
        data_frame, 280.0 * unit.kelvin, 315.0 * unit.kelvin
    )

    # Filter out long chain molecules, 3 + 4 membered rings
    # and 1, 3 carbonyl compounds where one of the carbonyls
    # is a ketone (cases where the enol form may be present in
    # non-negligible amounts).
    data_frame = filter_by_smirks(
        data_frame,
        None,
        [
            # 3 + 4 membered rings.
            "[#6r3]",
            "[#6r4]",
            # Long chain alkane /ether
            "[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]",
            # 1, 3 carbonyls with at least one ketone carbonyl.
            "[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](=[#8])-[#6]"
        ],
    )

    allowed_elements = ["H", "C", "O"]
    data_frame = filter_by_elements(data_frame, *allowed_elements)

    # Filter out any molecules with undefined stereochemistry
    data_frame = filter_undefined_stereochemistry(data_frame)

    return data_frame


@functools.lru_cache(3000)
def compute_component_finger_print(smiles, finger_print_type):

    oe_molecule = Molecule.from_smiles(smiles).to_openeye()

    finger_print = OEFingerPrint()
    OEMakeFP(finger_print, oe_molecule, finger_print_type)

    return finger_print


def compute_finger_print(mixture, finger_print_type):

    mixture_finger_print = tuple(
        compute_component_finger_print(x, finger_print_type) for x in mixture
    )

    return tuple(mixture_finger_print)


def compute_distance(mixture_a, mixture_b, finger_print_type):

    if mixture_a == mixture_b:
        return 0.0

    assert len(mixture_a) == len(mixture_b)

    finger_print_a = compute_finger_print(mixture_a, finger_print_type)
    finger_print_b = compute_finger_print(mixture_b, finger_print_type)

    distance = min(
        OETanimoto(finger_print_a[0], finger_print_b[0])
        + OETanimoto(finger_print_a[1], finger_print_b[1]),
        OETanimoto(finger_print_a[1], finger_print_b[0])
        + OETanimoto(finger_print_a[0], finger_print_b[1]),
    )

    return distance


def compute_distance_with_set(mixture_a, mixture_set, finger_print_type):

    open_list = [*mixture_set]
    distance = 0.0

    while len(open_list) > 0:

        most_similar = sorted(
            open_list, key=lambda x: compute_distance(mixture_a, x, finger_print_type)
        )[0]

        open_list.remove(most_similar)
        distance += compute_distance(mixture_a, most_similar, finger_print_type)

    return distance


def choose_molecules(
    property_of_interest,
    environment,
    finger_print_type,
    n_mixtures_per_environment,
    training_mixtures,
    root_output_directory,
):

    property_name = property_to_file_name(*property_of_interest)

    logger.info(f"{property_name}_{environment}: Starting.")

    try:

        data_frame = load_processed_data_set(
            os.path.join(
                "..",
                "..",
                "..",
                "data_availability",
                "data_by_environments",
                environment,
                "all_data",
            ),
            *property_of_interest,
        )

    except AssertionError:
        return

    # Filter out the training mixtures.
    data_frame = filter_by_substance_composition(data_frame, None, training_mixtures)
    data_frame = filter_data(data_frame)

    mixtures = {
        tuple(sorted((x["Component 1"], x["Component 2"])))
        for _, x in data_frame.iterrows()
    }

    open_list = [*mixtures]
    closed_list = []

    max_n_possible = min(len(open_list), n_mixtures_per_environment)

    while len(open_list) > 0 and len(closed_list) < n_mixtures_per_environment:

        def distance_metric(mixture):

            training_distance = compute_distance_with_set(
                mixture, training_mixtures, finger_print_type
            )
            test_distance = compute_distance_with_set(
                mixture, mixtures, finger_print_type
            )

            return training_distance + test_distance

        least_similar = sorted(open_list, key=distance_metric, reverse=True)[0]

        open_list.remove(least_similar)
        closed_list.append(least_similar)

        logger.info(
            f"{property_name}_{environment}: "
            f"{len(closed_list)} / {max_n_possible} selected"
        )

    if len(closed_list) == 0:
        return

    output_directory = os.path.join(root_output_directory, environment)
    os.makedirs(output_directory, exist_ok=True)

    file_path = os.path.join(output_directory, f"{property_name}")

    with open(f"{file_path}.json", "w") as file:
        json.dump(closed_list, file)

    smiles_to_pdf(closed_list, f"{file_path}.pdf")

    logger.info(f"{property_name}_{environment}: Finished.")


def main():

    logging.basicConfig(level=logging.INFO)

    root_output_directory = "test_sets"
    os.makedirs(root_output_directory, exist_ok=True)

    n_processes = 4

    # Define the types of property which are of interest.
    environments_of_interest = [
        "alcohol_ester",
        "alcohol_alkane",
        "ether_alkane",
        "ether_ketone",
        "alcohol_alcohol",
        "alcohol_ether",
        "alcohol_ketone",
        "ester_ester",
        "ester_alkane",
        "ester_ether",
        "ester_ketone",
        "alkane_alkane",
        "alkane_ketone",
        "ketone_ketone",
    ]

    n_mixtures_per_environment = 10

    properties_of_interest = [
        (EnthalpyOfMixing, SubstanceType.Binary),
        (Density, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
    ]

    finger_print_type = OEFPType_Tree

    training_mixtures = load_training_mixtures()

    with Pool(n_processes) as pool:

        partial_function = functools.partial(
            choose_molecules,
            finger_print_type=finger_print_type,
            n_mixtures_per_environment=n_mixtures_per_environment,
            training_mixtures=training_mixtures,
            root_output_directory=root_output_directory,
        )

        property_environments = []

        for property_of_interest in properties_of_interest:
            for environment in environments_of_interest:
                property_environments.append((property_of_interest, environment))

        list(pool.starmap(partial_function, property_environments))

    for property_of_interest in properties_of_interest:

        property_name = property_to_file_name(*property_of_interest)
        chosen_smiles = []

        for environment in environments_of_interest:

            data_path = os.path.join(
                root_output_directory, environment, f"{property_name}.json"
            )

            if not os.path.isfile(data_path):
                continue

            with open(data_path) as file:
                chosen_smiles.extend(json.load(file))

        chosen_smiles = [*{tuple(sorted(x)) for x in chosen_smiles}]

        with open(os.path.join(root_output_directory, f"{property_name}.json"), "w") as file:
            json.dump(chosen_smiles, file)

        smiles_to_pdf(
            chosen_smiles,
            os.path.join(root_output_directory, f"{property_name}.pdf"),
        )


if __name__ == "__main__":
    main()
