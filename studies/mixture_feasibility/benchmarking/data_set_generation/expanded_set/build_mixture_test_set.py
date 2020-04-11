"""
"""
import functools
import json
import logging
import os
from collections import defaultdict
from math import sqrt
from multiprocessing.pool import Pool
from tempfile import TemporaryDirectory

import numpy
import pandas
from evaluator import unit
from evaluator.properties import Density, EnthalpyOfMixing, ExcessMolarVolume
from openeye.oegraphsim import OEFingerPrint, OEFPType_Tree, OEMakeFP, OETanimoto
from openforcefield.topology import Molecule

from nistdataselection.curation.filtering import (
    filter_by_elements,
    filter_by_smirks,
    filter_by_substance_composition,
    filter_by_temperature,
    filter_undefined_stereochemistry,
)
from nistdataselection.curation.selection import StatePoint, select_data_points
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import property_to_file_name, smiles_to_pdf

logger = logging.getLogger(__name__)


def load_training_mixtures():
    """Loads in the training substances. These will
    be used to exclude training compounds from the test
    set.

    Returns
    -------
    set of tuple of str
        The substances in the training set.
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
    """Filters out data which is not of interest for the test
    set.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame to filter.

    Returns
    -------
    pandas.DataFrame
        The filtered data frame.
    """

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
            "[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](=[#8])-[#6]",
        ],
    )

    allowed_elements = ["H", "C", "O"]
    data_frame = filter_by_elements(data_frame, *allowed_elements)

    # Filter out any molecules with undefined stereochemistry
    data_frame = filter_undefined_stereochemistry(data_frame)

    return data_frame


@functools.lru_cache(3000)
def compute_component_finger_print(smiles, finger_print_type):
    """Computes the finger print for a particular molecule
    using the OpenEye toolkit.

    Parameters
    ----------
    smiles: str
        The smiles pattern to generate a finger print for.
    finger_print_type: OEFPTypeBase
        The type of finger print to generate.

    Returns
    -------
    OEFingerPrint
        The generate finger print.
    """

    oe_molecule = Molecule.from_smiles(smiles).to_openeye()

    finger_print = OEFingerPrint()
    OEMakeFP(finger_print, oe_molecule, finger_print_type)

    return finger_print


def compute_finger_print(mixture, finger_print_type):
    """Computes the finger print of a multicomponent
    substance.

    Parameters
    ----------
    mixture: tuple of str
        The smiles patterns which compose the substance.
    finger_print_type: OEFPTypeBase
        The type of finger print to generate.

    Returns
    -------
    tuple of OEFingerPrint
        The finger print of each component.
    """

    mixture_finger_print = tuple(
        compute_component_finger_print(x, finger_print_type) for x in mixture
    )

    return tuple(mixture_finger_print)


def compute_distance(mixture_a, mixture_b, finger_print_type):
    """Computes the 'distance' between two mixtures based on
    their finger prints.

    The distance is defined as the minimum of

    - the OETanimoto distance between component a of mixture a and
      component a of mixture b + the OETanimoto distance between
      component b of mixture a and component b of mixture b

    and

    - the OETanimoto distance between component b of mixture a and
      component a of mixture b + the OETanimoto distance between
      component a of mixture a and component b of mixture b

    Parameters
    ----------
    mixture_a: tuple of str
        The smiles patterns of the components in mixture a.
    mixture_b: tuple of str
        The smiles patterns of the components in mixture b.
    finger_print_type: OEFPTypeBase
        The type of finger print to base the distance metric
        on.

    Returns
    -------
    float
        The distance between the mixtures
    """

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
    """Computes the distance between a particular mixture
    and a set of mixtures.

    The distance is computed by:

    1. Computing the distance between `mixture_a` and each mixture
       in `mixture_set` using `compute_distance`
    2. Remove the mixture from `mixture_set` which is closest to
       `mixture_a` and adding the distance between the two to the
       total distance.
    3. Repeating step 2 until all mixtures have been removed from the
       `mixture_set`.

    Parameters
    ----------
    mixture_a: tuple of str
        The smiles patterns of the components in mixture a.
    mixture_set: list of tuple of str
        The smiles patterns of the components in the set of mixtures.
    finger_print_type: OEFPTypeBase
        The type of finger print to base the distance metric
        on.

    Returns
    -------
    float
        The distance between the mixtures
    """

    open_list = [*mixture_set]
    distance = 0.0

    while len(open_list) > 0:

        most_similar = sorted(
            open_list, key=lambda x: compute_distance(mixture_a, x, finger_print_type)
        )[0]

        open_list.remove(most_similar)
        distance += compute_distance(mixture_a, most_similar, finger_print_type)

    return distance


def choose_substances(
    property_of_interest,
    environment,
    finger_print_type,
    n_mixtures_per_environment,
    training_mixtures,
):
    """A function which aims to select a set of substances which are
    as distinct as possible from both the training and currently selected
    test set.

    This proceeds by:

    1. Selecting the molecule which is 'furthest' away from both the training
       set and the currently selected test set (which starts of empty), where
       the distance is defined as:

       sqrt(compute_distance_with_set(unselected_substance, training_set) ** 2 +
            compute_distance_with_set(unselected_substance, test_set) ** 2)

    2. Moving the selected molecule from the unselected set into the test set.

    3. Repeat steps 1 and two until either the target number of molecules have
       been selected, or there are no more unselected molecules to choose from.

    Parameters
    ----------
    property_of_interest: tuple of type of PhysicalProperty and SubstanceType
        The properties of interest.
    environment: str
        The environment (e.g. alcohol_alkane) to select molecules for.
    finger_print_type: OEFPTypeBase
        The type of finger print to base the distance metrics on.
    n_mixtures_per_environment: int
        The target number of molecules to select.
    training_mixtures: list of tuple of str
        The substances in the training set.
    Returns
    -------
    list of tuple of str
        The selected molecules.
    """

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
        return []

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

            return sqrt(training_distance ** 2 + test_distance ** 2)

        least_similar = sorted(open_list, key=distance_metric, reverse=True)[0]

        open_list.remove(least_similar)
        closed_list.append(least_similar)

        logger.info(
            f"{property_name}_{environment}: "
            f"{len(closed_list)} / {max_n_possible} selected"
        )

    return closed_list


def choose_data_points(
    property_of_interest, chosen_substances, target_states, environments_of_interest,
):
    """Select the data points to include in the benchmark set
    for each of the chosen substances.

    Parameters
    ----------
    property_of_interest: tuple of type of PhysicalProperty and SubstanceType
        The type of property to select data points for.
    chosen_substances: list of tuple of str
        The substances to choose data points for.
    target_states: list of StatePoint
        The target states to select data points at.

    Returns
    -------
    pandas.DataFrame
        The selected data points.
    """

    with TemporaryDirectory() as data_directory:

        data_frames = []

        for environment in environments_of_interest:

            data_folder = os.path.join(
                "..",
                "..",
                "..",
                "data_availability",
                "data_by_environments",
                environment,
                "all_data",
            )

            try:
                data_frame = load_processed_data_set(data_folder, *property_of_interest)
            except AssertionError:
                continue

            if len(data_frame) == 0:
                continue

            data_frames.append(data_frame)

        data_frame = pandas.concat(data_frames, ignore_index=True, sort=False)
        data_frame = filter_by_substance_composition(
            data_frame, chosen_substances, None
        )
        # Fill in the missing columns
        if "Exact Amount 1" not in data_frame:
            data_frame["Exact Amount 1"] = numpy.nan
        if "Exact Amount 2" not in data_frame:
            data_frame["Exact Amount 2"] = numpy.nan
        save_processed_data_set(data_directory, data_frame, *property_of_interest)

        target_states = {property_of_interest: target_states}

        selected_data_set = select_data_points(
            data_directory=data_directory,
            chosen_substances=None,
            target_state_points=target_states,
        )

    selected_data_frame = selected_data_set.to_pandas()

    # Prune any data points measured for too low or too high
    # mole fractions.
    selected_data_frame = selected_data_frame[
        (selected_data_frame["Mole Fraction 1"] > 0.15)
        & (selected_data_frame["Mole Fraction 1"] < 0.85)
    ]

    return selected_data_frame


def main():

    logging.basicConfig(level=logging.INFO)

    root_output_directory = "test_sets"
    os.makedirs(root_output_directory, exist_ok=True)

    n_processes = 4

    # Define the types of property which are of interest.
    properties_of_interest = [
        (EnthalpyOfMixing, SubstanceType.Binary),
        (Density, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
    ]

    # Define the state we would ideally chose data points at.
    target_states = [
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.25, 0.75)),
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.50, 0.50)),
        StatePoint(298.15 * unit.kelvin, 1.0 * unit.atmosphere, (0.75, 0.25)),
    ]

    # Define the environments of interest.
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

    # Chose the target number of substances per environment
    # / property.
    n_mixtures_per_environment = 10

    # Define the type of finger print to use for the distance
    # metric.
    finger_print_type = OEFPType_Tree

    # Load in the training substances so we can avoid selecting
    # them for the test set.
    training_mixtures = load_training_mixtures()

    # Select the particular substances to include in the test set.
    partial_function = functools.partial(
        choose_substances,
        finger_print_type=finger_print_type,
        n_mixtures_per_environment=n_mixtures_per_environment,
        training_mixtures=training_mixtures,
    )

    property_environments = []

    for property_of_interest in properties_of_interest:
        for environment in environments_of_interest:
            property_environments.append((property_of_interest, environment))

    with Pool(n_processes) as pool:

        selected_substances = list(
            pool.starmap(partial_function, property_environments)
        )

    # Combine the selected substance lists.
    substances_per_property = defaultdict(list)

    for substance_list, (property_type, _) in zip(
        selected_substances, property_environments
    ):
        substances_per_property[property_type].extend(substance_list)

    # Remove any duplicate substances
    substances_per_property = {x: [*set(y)] for x, y in substances_per_property.items()}

    for property_type, chosen_substances in substances_per_property.items():

        file_name = property_to_file_name(*property_type)
        file_path = os.path.join(root_output_directory, file_name)

        # Save the chosen substances to disk.
        smiles_to_pdf(chosen_substances, f"{file_path}.pdf")

        with open(f"{file_path}.json", "w") as file:
            json.dump(chosen_substances, file)

        # Select the data points.
        data_frame = choose_data_points(
            property_type,
            substances_per_property[property_type],
            target_states,
            environments_of_interest,
        )

        data_frame.to_csv(f"{file_path}.csv")


if __name__ == "__main__":
    main()
