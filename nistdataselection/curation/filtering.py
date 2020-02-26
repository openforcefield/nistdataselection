"""
Utilities for filtering data sets of measured physical properties.
"""
import logging

import numpy as np
from evaluator import unit
from openforcefield.topology import Molecule
from openforcefield.utils import UndefinedStereochemistryError

from nistdataselection.utils import analyse_functional_groups, find_smirks_matches

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
    data_set: pandas.DataFrame
        The data set to remove duplicates from.

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    if len(data_set) == 0:
        return data_set

    minimum_n_components = data_set["N Components"].min()
    maximum_n_components = data_set["N Components"].max()

    assert minimum_n_components == maximum_n_components

    data_set["Temperature (K)"] = data_set["Temperature (K)"].round(2)
    data_set["Pressure (kPa)"] = data_set["Pressure (kPa)"].round(3)

    subset_columns = ["Temperature (K)", "Pressure (kPa)", "Phase"]

    for index in range(minimum_n_components):

        data_set[f"Mole Fraction {index + 1}"] = data_set[
            f"Mole Fraction {index + 1}"
        ].round(6)

        subset_columns.extend(
            [
                f"Component {index + 1}",
                f"Role {index + 1}",
                f"Mole Fraction {index + 1}",
                f"Exact Amount {index + 1}",
            ]
        )

    subset_columns = [x for x in subset_columns if x in data_set]

    uncertainty_header = None

    for header in data_set:

        if "Uncertainty" not in header:
            continue

        assert uncertainty_header is None
        uncertainty_header = header

    if uncertainty_header in data_set:
        data_set = data_set.sort_values(uncertainty_header)

    return data_set.drop_duplicates(subset=subset_columns, keep="last")


def filter_by_temperature(data_set, min_temperature, max_temperature):
    """Filter the data set based on a minimum and maximum temperature.

    Parameters
    ----------
    data_set: pandas.DataFrame
        The data set to filter
    min_temperature : pint.Quantity
        The minimum temperature.
    max_temperature : pint.Quantity
        The maximum temperature.

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    min_temperature = min_temperature.to(unit.kelvin).magnitude
    max_temperature = max_temperature.to(unit.kelvin).magnitude

    return data_set[
        (min_temperature < data_set["Temperature (K)"])
        & (data_set["Temperature (K)"] < max_temperature)
    ]


def filter_by_pressure(data_set, min_pressure, max_pressure):
    """Filter the data set based on a minimum and maximum pressure.

    Parameters
    ----------
    data_set: pandas.DataFrame
        The data set to filter
    min_pressure : pint.Quantity
        The minimum pressure.
    max_pressure : pint.Quantity
        The maximum pressure.

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    min_pressure = min_pressure.to(unit.kilopascal).magnitude
    max_pressure = max_pressure.to(unit.kilopascal).magnitude

    return data_set[
        (min_pressure < data_set["Pressure (kPa)"])
        & (data_set["Pressure (kPa)"] < max_pressure)
    ]


def filter_by_elements(data_set, *allowed_elements):
    """Filters out those properties which were estimated for
     compounds which contain elements outside of those defined
     in `allowed_elements`.

    Parameters
    ----------
    data_set: pandas.DataFrame
        The data set to filter
    allowed_elements: str
        The symbols (e.g. C, H, Cl) of the elements to
        retain.

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    def filter_function(data_row):

        n_components = data_row["N Components"]

        for index in range(n_components):

            smiles = data_row[f"Component {index + 1}"]
            molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

            if not all([x.element.symbol in allowed_elements for x in molecule.atoms]):

                return False

        return True

    return data_set[data_set.apply(filter_function, axis=1)]


def filter_undefined_stereochemistry(data_set):
    """Filters out any substance which contain components with
    undefined stereochemistry.

    Parameters
    ----------
    data_set: pandas.DataFrame
        The data set to filter

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    def filter_function(data_row):

        n_components = data_row["N Components"]

        for index in range(n_components):

            smiles = data_row[f"Component {index + 1}"]

            try:
                Molecule.from_smiles(smiles)
            except UndefinedStereochemistryError:
                return False

        return True

    return data_set[data_set.apply(filter_function, axis=1)]


def filter_charged_molecules(data_set):
    """Filters out any molecules which have a net non-zero charge.

    Parameters
    ----------
    data_set: pandas.DataFrame
        The data set to filter

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    def filter_function(data_row):

        n_components = data_row["N Components"]

        for index in range(n_components):

            smiles = data_row[f"Component {index + 1}"]
            molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

            if np.isclose(sum([atom.formal_charge for atom in molecule.atoms]), 0.0):
                continue

            return False

        return True

    return data_set[data_set.apply(filter_function, axis=1)]


def filter_ionic_liquids(data_set):
    """Filters out ionic liquids.

    Parameters
    ----------
    data_set: pandas.DataFrame
        The data set to filter

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    def filter_function(data_row):

        n_components = data_row["N Components"]

        for index in range(n_components):

            smiles = data_row[f"Component {index + 1}"]

            if "." in smiles:
                return False

        return True

    return data_set[data_set.apply(filter_function, axis=1)]


def filter_property_by_value(data_set, property_type, minimum_value, maximum_value):
    """Filter out measured dielectric constants whose value is
    below a given threshold.

    Parameters
    ----------
    data_set: pandas.DataFrame
        The data set to filter
    property_type: type of PhysicalProperty
        The type of physical property to filter.
    minimum_value: unit.Quantity, optional
        The minimum acceptable value of the property.
    maximum_value: unit.Quantity, optional
        The maximum acceptable value of the property.

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    # noinspection PyUnresolvedReferences
    default_unit = property_type.default_unit()

    value_header = f"{property_type.__name__} Value ({default_unit:~})"
    assert value_header in data_set

    minimum_value = minimum_value.to(default_unit).magnitude
    maximum_value = maximum_value.to(default_unit).magnitude

    return data_set[
        (minimum_value < data_set[value_header])
        & (data_set[value_header] < maximum_value)
    ]


def filter_by_smiles(
    data_set, smiles_to_include, smiles_to_exclude, allow_partial_inclusion=False
):
    """Filters the data set so that it only contains either a specific set
    of smiles, or does not contain any of a set of specifically exluded smiles.

    Parameters
    ----------
    data_set: pandas.DataFrame
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

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    if (smiles_to_include is None and smiles_to_exclude is None) or (
        smiles_to_include is not None and smiles_to_exclude is not None
    ):

        raise ValueError(
            "Either a list of smiles to include, or a list of smiles to exclude must be provided, but not both."
        )

    if smiles_to_include is not None:
        smiles_to_exclude = []
    elif smiles_to_exclude is not None:
        smiles_to_include = []

    def filter_function(data_row):

        n_components = data_row["N Components"]

        component_smiles = [
            data_row[f"Component {index + 1}"] for index in range(n_components)
        ]

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

    return data_set[data_set.apply(filter_function, axis=1)]


def filter_by_smirks(data_set, smirks_to_include, smirks_to_exclude):
    """Filters a data set so that it only contains measurements made
    for molecules which contain (or don't) a set of chemical environments
    represented by SMIRKS patterns.

    Parameters
    ----------
    data_set: pandas.DataFrame
        The data set to filter
    smirks_to_include: list of str, optional
        The chemical environments which should be present.
        This option is mutually exclusive with `smirks_to_exclude`
    smirks_to_exclude: list of str, optional
        The chemical environments which should not be present.
        This option is mutually exclusive with `smirks_to_include`

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    if (smirks_to_include is None and smirks_to_exclude is None) or (
        smirks_to_include is not None and smirks_to_exclude is not None
    ):

        raise ValueError(
            "The `smiles_to_exclude` and `smirks_to_include` arguments are "
            "mutually exclusive."
        )

    if smirks_to_include is not None:
        smirks_to_exclude = []
    elif smirks_to_exclude is not None:
        smirks_to_include = []

    def filter_function(data_row):

        n_components = data_row["N Components"]

        component_smiles = [
            data_row[f"Component {index + 1}"] for index in range(n_components)
        ]

        inclusion_matches = find_smirks_matches(
            tuple(smirks_to_include), *component_smiles
        )
        exclusion_matches = find_smirks_matches(
            tuple(smirks_to_exclude), *component_smiles
        )

        if any(len(x) > 0 for x in inclusion_matches.values()):
            return True

        if any(len(x) > 0 for x in exclusion_matches.values()):
            return False

        return True

    return data_set[data_set.apply(filter_function, axis=1)]


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

    >>> filter_by_substance_composition(
    >>>     data_set,
    >>>     compositions_to_include=[
    >>>         ('CO',),
    >>>         ('C1=CC=CC=C1',),
    >>>         ('CCO', 'O')
    >>>     ]
    >>> )

    To excludes measurements made for an aqueous mix of benzene:

    >>> filter_by_substance_composition(
    >>>     data_set, compositions_to_exclude=[('O', 'C1=CC=CC=C1')]
    >>> )

    Parameters
    ----------
    data_set: pandas.DataFrame
        The data set to filter
    compositions_to_include: list of tuple of str, optional
        The substances compositions to retain, where each tuple in the list contains
        the smiles patterns which make up the substance to include. This option is mutually
        exclusive with `compositions_to_exclude`
    compositions_to_exclude: list of tuple str, optional
        The smiles patterns to exclude, where each tuple in the list contains
        the smiles patterns which make up the substance to enclude. This option is mutually
        exclusive with `compositions_to_include`

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
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

    def filter_function(data_row):

        n_components = data_row["N Components"]

        composition_tuple = tuple(
            sorted(
                [data_row[f"Component {index + 1}"] for index in range(n_components)]
            )
        )

        return (
            compositions_to_exclude is not None
            and composition_tuple not in validated_compositions_to_exclude
        ) or (
            compositions_to_include is not None
            and composition_tuple in validated_compositions_to_include
        )

    return data_set[data_set.apply(filter_function, axis=1)]


def filter_by_checkmol(data_set, *checkmol_codes):
    """Filters a set so that it only contains measurements made for substances which
    contain specific moieties. Moieties are defined by their `checkmol` codes.

    Parameters
    ----------
    data_set: pandas.DataFrame
        The data set to filter
    checkmol_codes: tuple of list of str
        The list of moieties to look for in the Nth component.
        Only one moiety from this list must be matched by the
        component for the measurement not to be filtered.

    Returns
    -------
    pandas.DataFrame
        The filtered data set.
    """

    n_components = data_set["N Components"].min()
    assert n_components == data_set["N Components"].max()

    assert len(checkmol_codes) == n_components

    def filter_function(data_row):

        component_smiles = [
            data_row[f"Component {index + 1}"] for index in range(n_components)
        ]
        component_moieties = [analyse_functional_groups(x) for x in component_smiles]

        if any(x is None for x in component_moieties):
            return False

        matched_code_sets = []

        for component_moiety_list in component_moieties:

            found_match = False

            for index in range(len(checkmol_codes)):

                if index in matched_code_sets:
                    continue

                if not any(x in checkmol_codes[index] for x in component_moiety_list):
                    continue

                found_match = True
                matched_code_sets.append(index)

                break

            if not found_match:
                return False

        return True

    return data_set[data_set.apply(filter_function, axis=1)]


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
    data_set: pandas.DataFrame
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

    current_number_of_properties = len(data_set)

    data_set = filter_duplicates(data_set)

    logger.debug(
        f"{current_number_of_properties - len(data_set)} duplicate data points were removed."
    )

    current_number_of_properties = len(data_set)

    data_set = filter_by_temperature(data_set, *temperature_range)
    logger.debug(
        f"{current_number_of_properties - len(data_set)} "
        f"data points were outside of the temperature range and were removed."
    )

    current_number_of_properties = len(data_set)

    data_set = filter_by_pressure(data_set, *pressure_range)
    logger.debug(
        f"{current_number_of_properties - len(data_set)} "
        f"data points were outside of the pressure range and were removed."
    )

    current_number_of_properties = len(data_set)

    data_set = filter_by_elements(data_set, *allowed_elements)
    logger.debug(
        f"{current_number_of_properties - len(data_set)} "
        f"data points were measured for substances containing unwanted elements and were removed."
    )

    # Make sure to only include molecules which have well defined stereochemistry
    current_number_of_properties = len(data_set)

    data_set = filter_undefined_stereochemistry(data_set)
    logger.debug(
        f"{current_number_of_properties - len(data_set)} "
        f"data points were measured for substances with undefined stereochemistry."
    )

    # Make sure to only include molecules which don't have a net charge.
    current_number_of_properties = len(data_set)

    data_set = filter_charged_molecules(data_set)
    logger.debug(
        f"{current_number_of_properties - len(data_set)} data points were measured "
        f"for substances with a net non-zero charge."
    )

    return data_set
