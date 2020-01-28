"""
Utilities for filtering data sets of measured physical properties.
"""
import logging
import math

import networkx
import numpy as np
from openforcefield.topology import Molecule
from openforcefield.utils import UndefinedStereochemistryError
from propertyestimator import unit

from nistdataselection.utils import standardize_smiles

logger = logging.getLogger(__name__)


def filter_duplicates(data_set):
    """Removes duplicate properties (i.e. those measured at the same
    states) from a data set. For now, the measurement with the largest
    reported experimental uncertainty is retained.

    Notes
    -----
    Temperatures are compared in kelvin to two decimal places, while
    pressures are compared in kilopascals to three decimal places.

    Warnings
    --------
    This method is not guaranteed to be deterministic.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to remove duplicates from.

    Returns
    -------
    PhysicalPropertyDataSet
        The processed data set.
    """

    properties_by_substance = {}

    for substance_id in data_set.properties:

        properties_by_substance[substance_id] = {}

        for physical_property in data_set.properties[substance_id]:

            property_type = physical_property.__class__.__name__

            # Partition the properties by type.
            if property_type not in properties_by_substance[substance_id]:
                properties_by_substance[substance_id][property_type] = {}

            # Partition the properties by state.
            temperature = physical_property.thermodynamic_state.temperature.to(
                unit.kelvin
            ).magnitude

            if physical_property.thermodynamic_state.pressure is None:

                state_tuple = (f"{temperature:.2f}", f"None")

            else:

                pressure = physical_property.thermodynamic_state.pressure.to(
                    unit.kilopascal
                ).magnitude
                state_tuple = (f"{temperature:.2f}", f"{pressure:.3f}")

            if state_tuple not in properties_by_substance[substance_id][property_type]:

                # Handle the easy case where this is the first time a
                # property at this state has been observed.
                properties_by_substance[substance_id][property_type][
                    state_tuple
                ] = physical_property
                continue

            existing_property = properties_by_substance[substance_id][property_type][
                state_tuple
            ]

            existing_uncertainty = (
                math.inf
                if existing_property.uncertainty is None
                else existing_property.uncertainty
            )
            current_uncertainty = (
                math.inf
                if physical_property.uncertainty is None
                else physical_property.uncertainty
            )

            base_unit = None

            if isinstance(existing_uncertainty, unit.Quantity):
                base_unit = existing_uncertainty.units

            elif isinstance(current_uncertainty, unit.Quantity):
                base_unit = current_uncertainty.units

            if base_unit is not None and isinstance(
                existing_uncertainty, unit.Quantity
            ):
                existing_uncertainty = existing_uncertainty.to(base_unit).magnitude

            if base_unit is not None and isinstance(current_uncertainty, unit.Quantity):
                current_uncertainty = current_uncertainty.to(base_unit).magnitude

            if (
                math.isinf(existing_uncertainty)
                and math.isinf(current_uncertainty)
                or existing_uncertainty < current_uncertainty
            ):

                # If neither property has an uncertainty, or the existing one has
                # a lower uncertainty keep that one.
                continue

            properties_by_substance[substance_id][property_type][
                state_tuple
            ] = physical_property

    # Rebuild the data set with only unique properties.
    unique_data_set = data_set.__class__()

    for substance_id in properties_by_substance:

        if substance_id not in unique_data_set.properties:
            unique_data_set.properties[substance_id] = []

        for property_type in properties_by_substance[substance_id]:

            for state_tuple in properties_by_substance[substance_id][property_type]:

                unique_data_set.properties[substance_id].append(
                    properties_by_substance[substance_id][property_type][state_tuple]
                )

    return unique_data_set


def filter_property_by_value(data_set, property_type, minimum_value, maximum_value):
    """Filter out measured dielectric constants whose value is
    below a given threshold.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    property_type: type of PhysicalProperty
        The type of physical property to filter.
    minimum_value: unit.Quantity, optional
        The minimum acceptable value of the property.
    maximum_value: unit.Quantity, optional
        The maximum acceptable value of the property.
    """

    if minimum_value is None and maximum_value is None:
        raise ValueError("Either a minimum or maximum value must be provided.")

    def filter_function(physical_property):

        if not isinstance(physical_property, property_type):
            return True

        if minimum_value is not None and physical_property.value < minimum_value:
            return False

        if maximum_value is not None and physical_property.value > maximum_value:
            return False

        return True

    data_set.filter_by_function(filter_function)


def filter_charged_molecules(data_set):
    """Filters out any molecules which have a net non-zero charge.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    """

    def filter_function(physical_property):

        for component in physical_property.substance.components:

            molecule = Molecule.from_smiles(component.smiles)

            if np.isclose(sum([atom.formal_charge for atom in molecule.atoms]), 0.0):
                continue

            return False

        return True

    data_set.filter_by_function(filter_function)


def filter_by_smiles(
    data_set, smiles_to_include, smiles_to_exclude, allow_partial_inclusion=False
):
    """Filters the data set so that it only contains either a specific set
    of smiles, or does not contain any of a set of specifically exluded smiles.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    smiles_to_include: list of str, optional
        The smiles patterns to retain. This option is mutually
        exclusive with `smiles_to_exclude`
    smiles_to_exclude: list of str, optional
        The smiles patterns to exclude. This option is mutually
        exclusive with `smiles_to_include`
    allow_partial_inclusion: bool
        If False, all the components in a substance must appear in
        the `smiles_to_include` list, otherwise, only some must appear.
        This option only applies when `smiles_to_include` is set.
    """

    if (smiles_to_include is None and smiles_to_exclude is None) or (
        smiles_to_include is not None and smiles_to_exclude is not None
    ):

        raise ValueError(
            "Either a list of smiles to include, or a list of smiles to exclude must be provided, but not both."
        )

    if smiles_to_include is not None:
        smiles_to_include = standardize_smiles(*smiles_to_include)
        smiles_to_exclude = []
    elif smiles_to_exclude is not None:
        smiles_to_exclude = standardize_smiles(*smiles_to_exclude)
        smiles_to_include = []

    def filter_function(physical_property):

        component_smiles = [x.smiles for x in physical_property.substance.components]
        component_smiles = standardize_smiles(*component_smiles)

        if any(x in smiles_to_exclude for x in component_smiles):
            return False

        if not allow_partial_inclusion and not all(
            x in smiles_to_include for x in component_smiles
        ):
            return False

        if allow_partial_inclusion and not any(
            x in smiles_to_include for x in component_smiles
        ):
            return False

        return True

    data_set.filter_by_function(filter_function)


def filter_by_substance_composition(
    data_set, compositions_to_include, compositions_to_exclude
):
    """Filters the data set so that it only contains properties measured for substances
    of specified compositions.

    This method is similar to `filter_by_smiles`, however here we explicitly define
    the full substances compositions, rather than individual smiles which should
    either be included or excluded.

    Examples
    --------
    To filter the data set to only include measurements for pure methanol, pure benzene
    or a aqueous ethanol mix:

    >>> # Load in the data set of properties which will be used for comparisons
    >>> from propertyestimator.datasets import ThermoMLDataSet
    >>> data_set = ThermoMLDataSet.from_doi('DOI')
    >>>
    >>> filter_by_substance_composition(compositions_to_include=[('CO',),
    >>>                                                          ('C1=CC=CC=C1',),
    >>>                                                          ('CCO', 'O')])

    To excludes measurements made for an aqueous mix of benzene:

    >>> # Load in the data set of properties which will be used for comparisons
    >>> from propertyestimator.datasets import ThermoMLDataSet
    >>> data_set = ThermoMLDataSet.from_doi('DOI')
    >>>
    >>> filter_by_substance_composition(compositions_to_exclude=[('O', 'C1=CC=CC=C1')])

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    compositions_to_include: list of tuple of str, optional
        The substances compositions to retain, where each tuple in the list contains
        the smiles patterns which make up the substance to include. This option is mutually
        exclusive with `compositions_to_exclude`
    compositions_to_exclude: list of tuple str, optional
        The smiles patterns to exclude, where each tuple in the list contains
        the smiles patterns which make up the substance to enclude. This option is mutually
        exclusive with `compositions_to_include`
    """

    if (compositions_to_include is None and compositions_to_exclude is None) or (
        compositions_to_include is not None and compositions_to_exclude is not None
    ):

        raise ValueError(
            "Either a list of compositions to include, or a list of compositions "
            "to exclude must be provided, but not both."
        )

    # Make sure the provided smiles tuple lists are in an expected format.
    validated_compositions_to_include = []
    validated_compositions_to_exclude = []

    for smiles_tuple in (
        [] if compositions_to_include is None else compositions_to_include
    ):

        if not isinstance(smiles_tuple, (str, tuple)):

            raise ValueError(
                "The `compositions_to_include` argument must either be a list of strings, "
                "or a list of string tuples"
            )

        smiles_list = (
            [smiles_tuple] if isinstance(smiles_tuple, str) else [*smiles_tuple]
        )
        validated_compositions_to_include.append(tuple(sorted(smiles_list)))

    for smiles_tuple in (
        [] if compositions_to_exclude is None else compositions_to_exclude
    ):

        if not isinstance(smiles_tuple, (str, tuple)):
            raise ValueError(
                "The `compositions_to_exclude` argument must either be a list of strings, "
                "or a list of string tuples"
            )

        smiles_list = (
            [smiles_tuple] if isinstance(smiles_tuple, str) else [*smiles_tuple]
        )
        validated_compositions_to_exclude.append(tuple(sorted(smiles_list)))

    def filter_function(physical_property):

        composition_tuple = tuple(
            sorted(
                [
                    component.smiles
                    for component in physical_property.substance.components
                ]
            )
        )

        return (
            compositions_to_exclude is not None
            and composition_tuple not in validated_compositions_to_exclude
        ) or (
            compositions_to_include is not None
            and composition_tuple in validated_compositions_to_include
        )

    data_set.filter_by_function(filter_function)


def filter_undefined_stereochemistry(data_set):
    """Filters out any substance which contain components with
    undefined stereochemistry.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    """

    def filter_function(physical_property):

        for component in physical_property.substance.components:

            try:
                Molecule.from_smiles(component.smiles)
            except UndefinedStereochemistryError:
                return False

        return True

    data_set.filter_by_function(filter_function)


def filter_ionic_liquids(data_set):
    """Filters out ionic liquids.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    """

    def filter_function(physical_property):

        for component in physical_property.substance.components:

            if "." in component.smiles:
                return False

        return True

    data_set.filter_by_function(filter_function)


def filter_by_number_of_halogens(data_set, minimum_halogens=0, maximum_halogens=3):
    """Filters the data set so that it only contains substance whose
    components contain a number of halogens within the specified limits.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    minimum_halogens: int
        The inclusive minimum number of halogens a component
        must have.
    maximum_halogens: int
        The inclusive maximum number of halogens a component
        may have
    """
    halogens = ["F", "Cl", "Br", "I"]

    def filter_function(physical_property):

        for component in physical_property.substance.components:

            molecule = Molecule.from_smiles(component.smiles)
            molecule_halogens = (
                atom for atom in molecule.atoms if atom.element.symbol in halogens
            )

            number_of_halogens = {halogen: 0 for halogen in halogens}

            for atom in molecule_halogens:
                number_of_halogens[atom.element.symbol] += 1

            for halogen_count in number_of_halogens.values():

                if halogen_count < minimum_halogens or halogen_count > maximum_halogens:
                    return False

        return True

    data_set.filter_by_function(filter_function)


def filter_by_longest_path_length(
    data_set, maximum_path_length, include_ring_containing=False
):
    """Filters the data set so that it only contains substance whose
    components are shorter than the maximum path length (ignoring any
    hydrogens). In simple chain molecules, this would be the length of
    the chain e.g `maximum_path_length=5` would retain alkanes up to
    and including pentane.

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter
    maximum_path_length: int
        The maximum eccentricity in the molecular graph, excluding
        any hydrogens.
    include_ring_containing: bool
        Determines whether this should be applied to substances containing
        rings, such as aromatics. When included, this behaviour may not be
        well defined.
    """

    def filter_function(physical_property):

        for component in physical_property.substance.components:

            molecule = Molecule.from_smiles(component.smiles)
            molecule_graph = molecule.to_networkx()

            if include_ring_containing is False:

                try:
                    networkx.find_cycle(molecule_graph)
                except networkx.NetworkXNoCycle:
                    pass
                else:
                    continue

            hydrogen_nodes = reversed(
                sorted(
                    [
                        node
                        for node, data in molecule_graph.nodes(data=True)
                        if data["atomic_number"] == 1
                    ]
                )
            )

            for node in hydrogen_nodes:
                molecule_graph.remove_node(node)

            diameter = networkx.diameter(molecule_graph) + 1

            if diameter <= maximum_path_length:
                continue

            return False

        return True

    data_set.filter_by_function(filter_function)


def apply_standard_filters(
    data_set, temperature_range, pressure_range, allowed_elements
):
    """Filters the data set (in the listed order) such as to remove:

    * any duplicate properties (`filter_duplicates`).
    * data points measured outside of the allowed `temperature_range` (`data_set.filter_by_temperature`).
    * data points measured outside of the allowed `pressure_range` (`data_set.filter_by_pressure`).
    * data points measured for substance which contain elements not in the `allowed_elements` list
      (`data_set.filter_by_elements`).
    * remove any substances of undefined stereochemistry (`filter_undefined_stereochemistry`).
    * remove any substance with a net non-zero charge (`filter_charged_molecules`).

    Parameters
    ----------
    data_set: PhysicalPropertyDataSet
        The data set to filter.
    temperature_range: tuple of unit.Quantity and unit.Quantity
        The minimum and maximum temperatures.
    pressure_range: tuple of unit.Quantity and unit.Quantity
        The minimum and maximum pressures.
    allowed_elements: list of str
        A list of the allowed atomic elements.

    Returns
    -------
    PhysicalPropertyDataSet
        The filtered data set.
    """

    current_number_of_properties = data_set.number_of_properties

    data_set = filter_duplicates(data_set)
    logger.info(
        f"{current_number_of_properties - data_set.number_of_properties} duplicate data points were removed."
    )

    current_number_of_properties = data_set.number_of_properties

    data_set.filter_by_temperature(*temperature_range)
    logger.info(
        f"{current_number_of_properties - data_set.number_of_properties} "
        f"data points were outside of the temperature range and were removed."
    )

    current_number_of_properties = data_set.number_of_properties

    data_set.filter_by_pressure(*pressure_range)
    logger.info(
        f"{current_number_of_properties - data_set.number_of_properties} "
        f"data points were outside of the pressure range and were removed."
    )

    current_number_of_properties = data_set.number_of_properties

    data_set.filter_by_elements(*allowed_elements)
    logger.info(
        f"{current_number_of_properties - data_set.number_of_properties} "
        f"data points were measured for substances containing unwanted elements and were removed."
    )

    # Make sure to only include molecules which have well defined stereochemistry
    current_number_of_properties = data_set.number_of_properties

    filter_undefined_stereochemistry(data_set)
    logger.info(
        f"{current_number_of_properties - data_set.number_of_properties} "
        f"data points were measured for substances with undefined stereochemistry."
    )

    # Make sure to only include molecules which don't have a net charge.
    current_number_of_properties = data_set.number_of_properties

    filter_charged_molecules(data_set)
    logger.info(
        f"{current_number_of_properties - data_set.number_of_properties} data points were measured "
        f"for substances with a net non-zero charge."
    )

    return data_set
