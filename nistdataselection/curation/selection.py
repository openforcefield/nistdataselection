"""
Tools for selecting sets of molecules and corresponding measured physical property
data points for use in optimizing and benchmarking molecular force fields.
"""
import functools
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Tuple, Type

from evaluator import unit
from evaluator.datasets import PhysicalProperty, PhysicalPropertyDataSet
from nistdataselection.processing import load_processed_data_set
from nistdataselection.utils.utils import (
    SubstanceType,
    find_parameter_smirks_matches,
    find_smirks_matches,
    get_atom_count,
    invert_dict_of_iterable,
    property_to_type_tuple,
)

logger = logging.getLogger(__name__)


@dataclass
class SubstanceData:
    """Represents all of the data needed to rank how preferential
    it would be to choose a specific substance.
    """

    substance_tuple: Tuple[str]
    smirks_exercised: Set[str]
    property_types: Set[Tuple[Type[PhysicalProperty], SubstanceType]]


class StatePoint:
    """Represents the thermodynamic state and the substance composition
    at which a measurement was made.
    """

    def __init__(self, temperature, pressure, mole_fractions):
        """Constructs a new `StatePoint` object.

        Parameters
        ----------
        temperature: unit.Quantity
            The temperature that the measurement was recorded at.
        pressure: unit.Quantity
            The pressure that the measurement was recorded at.
        mole_fractions: tuple of float
            The mole fractions of each of the components.
        """
        self.temperature = temperature
        self.pressure = pressure

        self.mole_fractions = mole_fractions

    @classmethod
    def from_physical_property(cls, physical_property):
        """Constructs a new `StatePoint` from an existing
        `PhysicalProperty`.

        Parameters
        ----------
        physical_property: PhysicalProperty
            The property to extract the state from.

        Returns
        -------
        StatePoint
            The extracted state.
        """
        mole_fractions = tuple(
            [
                next(iter(physical_property.substance.get_amounts(component))).value
                for component in physical_property.substance.components
            ]
        )

        return cls(
            physical_property.thermodynamic_state.temperature,
            physical_property.thermodynamic_state.pressure,
            mole_fractions,
        )

    @staticmethod
    def individual_distances(state_a, state_b):
        """Defines a metric for how close a this state point is to another
        based on the distances of the individual components.

        Parameters
        ----------
        state_a: StatePoint
            The first state point to compare.
        state_b: StatePoint
            The second state point to compare.

        Returns
        -------
        float
            difference in mole fractions ^ 2
        float
            difference in pressure in kPa ^ 2,
        float
            difference in temperature in K ^ 2
        """

        state_a_temperature = round(state_a.temperature.to(unit.kelvin).magnitude, 2)
        state_b_temperature = round(state_b.temperature.to(unit.kelvin).magnitude, 2)

        temperature_distance_sqr = (state_b_temperature - state_a_temperature) ** 2

        assert (state_a.pressure is None and state_b.pressure is None) or (
            state_a.pressure is not None and state_b.pressure is not None
        )

        pressure_distance_sqr = 0.0

        if state_a.pressure is not None:

            state_a_pressure = round(state_a.pressure.to(unit.kilopascal).magnitude, 3)
            state_b_pressure = round(state_b.pressure.to(unit.kilopascal).magnitude, 3)

            pressure_distance_sqr = (state_b_pressure - state_a_pressure) ** 2

        assert len(state_a.mole_fractions) == len(state_b.mole_fractions)

        mole_fraction_distance_sqr = sum(
            [
                (state_b.mole_fractions[i] - state_a.mole_fractions[i]) ** 2
                for i in range(len(state_a.mole_fractions))
            ]
        )

        distance_tuple = (
            mole_fraction_distance_sqr,
            pressure_distance_sqr,
            temperature_distance_sqr,
        )

        return distance_tuple

    @staticmethod
    def distance(state_a, state_b):
        """Defines a metric for how close one state point is to another.
        This is based off of the `individual_distances` metric.

        Parameters
        ----------
        state_a: StatePoint
            The first state point to compare.
        state_b: StatePoint
            The second state point to compare.

        Returns
        -------
        float
            The distance to the other state point.
        """

        return math.sqrt(sum(StatePoint.individual_distances(state_a, state_b)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return self != other

    def __hash__(self):

        pressure_string = ""

        if self.pressure is not None:
            pressure_string = f"{self.pressure.to(unit.kilopascal).magnitude:.3f}"

        return hash(
            (
                pressure_string,
                f"{self.temperature.to(unit.kelvin).magnitude:.2f}",
                self.mole_fractions,
            )
        )

    def __repr__(self):

        if self.pressure is None:
            return (
                f"T={self.temperature:~} x=({', '.join(map(str, self.mole_fractions))})"
            )

        return f"T={self.temperature:~} P={self.pressure:~} x=({', '.join(map(str, self.mole_fractions))})"


def _build_substance_data(
    data_directory, target_substances_per_property, smirks_to_exercise
):
    """Loads all of the different data sets for each property type of
    interest and converts them into a single list of `SubstanceData`
    objects.

    Any substances which don't exercise at least one of the chemical
    environments of interest are ignored.

    Parameters
    ----------
    data_directory: str
        The directory which contains the processed pandas
        data sets
    target_substances_per_property: dict of tuple of type and SubstanceType and int
        The target number of unique substances to choose for each
        type of property of interest.
    smirks_to_exercise: list of str
        A list of those smirks patterns which represent those chemical environments
         which we to aim to exercise.

    Returns
    -------
    list of SubstanceData
        The loaded substance data.
    """
    all_substance_tuples = defaultdict(set)
    all_smiles_patterns = set()

    for property_type, substance_type in target_substances_per_property:

        # Load the full data sets from the processed data file
        data_set = load_processed_data_set(
            data_directory, property_type, substance_type
        )

        for substance in data_set.substances:

            # Extract all of the components of the substance as smiles patterns.
            substance_smiles = [component.smiles for component in substance.components]
            substance_tuple = tuple(sorted(substance_smiles))

            all_substance_tuples[substance_tuple].add((property_type, substance_type))
            all_smiles_patterns.update(substance_tuple)

    # Build the list of substances which we can choose from
    all_substance_data = []

    for substance_tuple in all_substance_tuples:

        # Make sure that this smiles tuple does actually exercise at least one
        # of the chemical environments of interest.
        smiles_per_smirks = find_smirks_matches(
            tuple(smirks_to_exercise), *substance_tuple
        )
        all_exercised_smirks = set(
            [smirks for smirks, smiles in smiles_per_smirks.items() if len(smiles) > 0]
        )

        smirks_per_smiles = invert_dict_of_iterable(smiles_per_smirks)

        exercised_smirks_of_interest = set()

        for smiles_pattern in substance_tuple:

            if (
                smiles_pattern not in smirks_per_smiles
                or len(smirks_per_smiles[smiles_pattern]) == 0
            ):
                continue

            exercised_smirks_of_interest.update(smirks_per_smiles[smiles_pattern])

        if len(exercised_smirks_of_interest) == 0:
            continue

        substance_data = SubstanceData(
            substance_tuple=substance_tuple,
            smirks_exercised=all_exercised_smirks,
            property_types=all_substance_tuples[substance_tuple],
        )

        all_substance_data.append(substance_data)

    return all_substance_data


def _filter_by_target_substances(
    substance_data, target_substances_per_property, substances_chosen_per_property
):
    """Filters substances based on whether this substance
    moves us closer towards the target number of substances
    or not.

    Parameters
    ----------
    target_substances_per_property: dict of tuple of type and SubstanceType and int
        The target number of unique substances to choose for each
        type of property of interest.
    substances_chosen_per_property: dict of tuple of type and SubstanceType and int
        The currently chosen number of unique substances which exercise
        each type of property of interest.

    Returns
    -------
    bool
        Whether to retain this substance or not.
    """

    increases_target_substance_count = False

    for property_tuple in substance_data.property_types:

        if (
            substances_chosen_per_property[property_tuple] + 1
            > target_substances_per_property[property_tuple]
        ):
            continue

        increases_target_substance_count = True
        break

    return increases_target_substance_count


def rank_substance_data(
    substance_data,
    target_substances_per_property,
    substances_chosen_per_property,
    substances_chosen_per_smirks,
):
    """A function used to rank a list of substances tuples in order
    of preference to include them in the chosen data set.

    Parameters
    ----------
    substance_data: SubstanceData
        The information about a given substance to assign
        ranking information to.
    target_substances_per_property: dict of tuple of type and SubstanceType and int
        The target number of unique substances to choose for each
        type of property of interest.
    substances_chosen_per_property: dict of tuple of type and SubstanceType and int
        The currently chosen number of unique substances which exercise
        each type of property of interest.
    substances_chosen_per_smirks: dict of str and int
        The currently chosen number of unique substances which exercise
        each chemical environment of interest.

    Returns
    -------
    int
        The number of least exercised smirks which would be exercised
        if this substance was included in the set.
    float
        The inverse total number of atoms in the substance.
    int
        The number of property types for which this substance has data.
    """

    # We are interested in those substances which have data for
    # multiple of the different properties we are interested in.
    # We sort by the number of missing property types as substances
    # will be sorted according to smallest to largest.
    number_of_property_types = len(substance_data.property_types)

    # Come up with a ranking which will prioritise those molecules which
    # exercise those smirks which are currently least exercised.
    smirks_per_number_of_chosen = defaultdict(list)

    for smirks, count in substances_chosen_per_smirks.items():
        smirks_per_number_of_chosen[count].append(smirks)

    lowest_smirks_count = min(substances_chosen_per_smirks.values())
    number_of_least_exercised_smirks = 0

    number_of_least_exercised_smirks += sum(
        [
            int(smirks in smirks_per_number_of_chosen[lowest_smirks_count])
            for smirks in substance_data.smirks_exercised
        ]
    )

    # We prefer smaller molecules as they will be quicker to simulate
    # and their properties should converge faster (compared to larger,
    # possibly more flexible molecule with more degrees of freedom to sample).
    inverse_number_of_atoms = 1.0 / sum(
        [get_atom_count(smiles) for smiles in substance_data.substance_tuple]
    )

    # Return the tuple to sort by
    return_tuple = (
        number_of_least_exercised_smirks,
        number_of_property_types,
        inverse_number_of_atoms,
    )

    return return_tuple


def select_substances(
    data_directory,
    target_substances_per_property,
    smirks_to_exercise=None,
    ranking_function=None,
):
    """Selects the minimum set of molecules which (if possible) have a target amount
     of data for a number of properties of interest, and which exercise a specified
     region of chemical space represented by a series of smirks patterns.

    Parameters
    ----------
    data_directory: str
        The directory which contains the processed pandas
        data sets
    target_substances_per_property: dict of tuple of type and SubstanceType and int
        The target number of unique substances to choose for each
        type of property of interest.
    smirks_to_exercise: list of str, optional
        A list of those smirks patterns which represent the different
        chemical environments we wish to exercise. If None, all VdW
        parameters in the OpenFF Parsley force field will be targeted.
    ranking_function: function, optional
        The function to use when ranking each substance for selection. If None,
        the built in `rank_substance_data` function is used.

    Returns
    -------
    list of tuple of str
        The smiles representations of the chosen substances.
    """

    # Fill the VdW smirks to exercise list if None is passed
    if smirks_to_exercise is None:
        smirks_to_exercise = list(set(find_parameter_smirks_matches("vdW").keys()))

    if ranking_function is None:
        ranking_function = rank_substance_data

    # Build a list of all of the substances from the different data sets.
    logger.info("Building the substance data lists.")
    substance_data_open_list = _build_substance_data(
        data_directory, target_substances_per_property, smirks_to_exercise
    )

    # Select substances from the open list until there are no more substances
    # to choose from, or we have hit the target number of substances per
    # property.
    logger.info("Starting to choose substances.")
    chosen_substances = []

    substances_chosen_per_property = defaultdict(int)
    substances_chosen_per_smirks = {smirks: 0 for smirks in smirks_to_exercise}

    while len(substance_data_open_list) > 0:

        # Pick the first substance from the list which has been sorted
        # according to the ranking function, provided the list still
        # has substances to choose from.
        ranked_substance_data = sorted(
            substance_data_open_list,
            key=functools.partial(
                ranking_function,
                target_substances_per_property=target_substances_per_property,
                substances_chosen_per_property=substances_chosen_per_property,
                substances_chosen_per_smirks=substances_chosen_per_smirks,
            ),
            reverse=True,
        )

        chosen_substance_data = ranked_substance_data[0]

        # Move the substance from the open to the closed list.
        substance_data_open_list.remove(chosen_substance_data)
        chosen_substances.append(chosen_substance_data)

        # Update the bookkeeping counts
        for property_tuple in chosen_substance_data.property_types:
            substances_chosen_per_property[property_tuple] += 1

        for smirks_pattern in chosen_substance_data.smirks_exercised:

            if smirks_pattern not in substances_chosen_per_smirks:
                continue

            substances_chosen_per_smirks[smirks_pattern] += 1

        # Filter the remaining substances based on the target counts.
        substance_data_open_list = list(
            filter(
                functools.partial(
                    _filter_by_target_substances,
                    target_substances_per_property=target_substances_per_property,
                    substances_chosen_per_property=substances_chosen_per_property,
                ),
                substance_data_open_list,
            )
        )

    chosen_substances = [
        substance_data.substance_tuple for substance_data in chosen_substances
    ]
    return chosen_substances


def _cluster_properties_around_states(physical_properties, target_state_points):
    """Clusters a set of physical properties around a set
    of target state points using the distance metric defined
    by `_state_distance`.

    Parameters
    ----------
    physical_properties: list of PhysicalProperty
        The list of properties to cluster.
    target_state_points: dict of tuple of type and SubstanceType and list of StatePoint
        A list of the state points to cluster the data around.

    Returns
    -------
    dict of property tuple and dict of StatePoint and list of PhysicalProperty
        The clustered properties stored in a dictionary partitioned first by
        property type, and then by target state.
    """

    clusters = defaultdict(list)

    for physical_property in physical_properties:

        property_tuple = property_to_type_tuple(physical_property)
        state_point = StatePoint.from_physical_property(physical_property)

        if property_tuple not in target_state_points:
            continue

        closest_cluster_state = None
        shortest_cluster_distance = sys.float_info.max

        for target_state_point in target_state_points[property_tuple]:

            distance = StatePoint.distance(state_point, target_state_point)

            if distance >= shortest_cluster_distance:
                continue

            closest_cluster_state = target_state_point
            shortest_cluster_distance = distance

        clusters[closest_cluster_state].append(physical_property)

    return clusters


def select_data_points(data_directory, chosen_substances, target_state_points):
    """The method attempts to find a set of data points for each
    property which are clustered around the set of conditions specified
    in the `target_state_points` input array.

    The points will be chosen so as to try and maximise the number of
    properties measured at the same condition (e.g. ideally we would
    have a data point for each property at T=298.15 and p=1atm) as this
    will maximise the chances that we can extract all properties from a
    single simulation.

    Parameters
    ----------
    data_directory: str
        The directory which contains the processed pandas
        data sets
    chosen_substances: list of tuple of str
        The substances to choose data points for.
    target_state_points: dict of tuple of type and SubstanceType and list of StatePoint
        A list of the state points for which we would ideally have data
        points for. The value tuple should be of the form
        (temperature, pressure, (mole fraction 0, ..., mole fraction N))

    Returns
    -------
    PhysicalPropertyDataSet
        A data set which contains the chosen data points.
    """

    # Load the full data set from the processed data files
    data_set = PhysicalPropertyDataSet()

    for property_type, substance_type in target_state_points:

        property_data_set = load_processed_data_set(
            data_directory, property_type, substance_type
        )
        data_set.merge(property_data_set)

    properties_by_substance = defaultdict(list)

    # Partition the properties by their substance components,
    # filtering out any not chosen substances.
    for substance in data_set.substances:

        substance_tuple = tuple(
            sorted([component.smiles for component in substance.components])
        )

        if substance_tuple not in chosen_substances:
            continue

        properties_by_substance[substance_tuple].extend(
            data_set.properties_by_substance(substance)
        )

    # Start to choose the state points.
    return_data_set = PhysicalPropertyDataSet()

    for substance_tuple in properties_by_substance:

        # Cluster the data points around the closest states of interest.
        clustered_properties = _cluster_properties_around_states(
            properties_by_substance[substance_tuple], target_state_points
        )

        # For each cluster, we try to find the state points for which we have
        # measured the most types of properties (i.e. prioritise states
        # for which we have a density, dielectric and enthalpy measurement
        # over those for which we only have a density measurement).
        for target_state_point, physical_properties in clustered_properties.items():

            properties_per_state = defaultdict(list)
            property_types_per_state = defaultdict(set)

            # Refactor the properties into more convenient data structures.
            for physical_property in physical_properties:

                state_point = StatePoint.from_physical_property(physical_property)
                property_tuple = property_to_type_tuple(physical_property)

                properties_per_state[state_point].append(physical_property)
                property_types_per_state[state_point].add(property_tuple)

            # Sort the state points based on their distance to the target state.
            sorted_states_points = list(
                sorted(
                    properties_per_state.keys(),
                    key=functools.partial(
                        StatePoint.individual_distances, target_state_point
                    ),
                )
            )

            # Keep track of the properties which we need to choose a state point for
            properties_to_cover = set(
                property_tuple for property_tuple in target_state_points
            )
            # as well as the chosen state points
            chosen_state_points = set()

            # Iteratively consider state points which have all data points, down
            # to state points for which we only have single property measurements.
            for target_number_of_properties in reversed(
                range(1, len(target_state_points) + 1)
            ):

                for state_point in sorted_states_points:

                    property_types_at_state = property_types_per_state[state_point]

                    if len(property_types_at_state) != target_number_of_properties:
                        continue

                    if (
                        len(properties_to_cover.intersection(property_types_at_state))
                        == 0
                    ):
                        continue

                    chosen_state_points.add(state_point)

                    properties_to_cover = properties_to_cover.symmetric_difference(
                        properties_to_cover.intersection(property_types_at_state)
                    )

            # Add the properties which were measured at the chosen state points
            # to the returned data set.
            for state_point in chosen_state_points:

                if len(properties_per_state[state_point]) == 0:
                    continue

                return_data_set.add_properties(*properties_per_state[state_point])

    return return_data_set
