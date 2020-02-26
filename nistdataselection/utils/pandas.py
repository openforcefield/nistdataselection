"""
A utility for going between `evaluator.datasets.PhysicalPropertyDataSet
objects and `pandas.DataFrame` objects.
"""

import evaluator.properties
import numpy
from evaluator import unit
from evaluator.datasets import MeasurementSource, PhysicalPropertyDataSet, PropertyPhase
from evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from evaluator.thermodynamics import ThermodynamicState


def data_set_from_data_frame(data_frame):
    """Converts a `pandas.DataFrame` to a `PhysicalPropertyDataSet` object.
    See the `PhysicalPropertyDataSet.to_pandas()` function for information
    on the required columns.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame to convert.

    Returns
    -------
    PhysicalPropertyDataSet
        The converted data set.
    """

    return_value = PhysicalPropertyDataSet()

    if len(data_frame) == 0:
        return return_value

    # Make sure the base columns are present.
    required_base_columns = [
        "Temperature (K)",
        "Pressure (kPa)",
        "Phase",
        "N Components",
        "Source",
    ]

    assert all(x in data_frame for x in required_base_columns)

    # Make sure the substance columns are present.
    max_components = max(int(x) for x in data_frame["N Components"])
    assert max_components > 0

    required_components_columns = [
        x
        for i in range(max_components)
        for x in [
            f"Component {i + 1}",
            f"Role {i + 1}",
            f"Mole Fraction {i + 1}",
            f"Exact Amount {i + 1}",
        ]
    ]

    assert all(x in data_frame for x in required_components_columns)

    property_types = []

    for column_name in data_frame:

        if " Value" not in column_name:
            continue

        column_name_split = column_name.split(" ")

        assert len(column_name_split) >= 2

        property_type = getattr(evaluator.properties, column_name_split[0])
        property_types.append(property_type)

    assert len(property_types) > 0

    # Make sure we don't have duplicate property columns.
    assert len(set(property_types)) == len(property_types)

    properties = []

    for _, row in data_frame.iterrows():

        # Create the substance from the component columns
        number_of_components = row["N Components"]

        substance = Substance()

        for component_index in range(number_of_components):

            smiles = row[f"Component {component_index + 1}"]
            role = Component.Role[row[f"Role {component_index + 1}"]]
            mole_fraction = row[f"Mole Fraction {component_index + 1}"]
            exact_amount = row[f"Exact Amount {component_index + 1}"]

            assert not numpy.isnan(mole_fraction) or not numpy.isnan(exact_amount)

            component = Component(smiles, role)

            if not numpy.isnan(mole_fraction):
                substance.add_component(component, MoleFraction(mole_fraction))
            if not numpy.isnan(exact_amount):
                substance.add_component(component, ExactAmount(exact_amount))

        # Extract the state
        pressure = row["Pressure (kPa)"] * unit.kilopascal
        temperature = row["Temperature (K)"] * unit.kelvin

        thermodynamic_state = ThermodynamicState(temperature, pressure)

        phase = PropertyPhase.from_string(row["Phase"])

        source = MeasurementSource(reference=row["Source"])

        for property_type in property_types:

            default_unit = property_type.default_unit()
            value_header = f"{property_type.__name__} Value ({default_unit:~})"

            if numpy.isnan(row[value_header]):
                continue

            value = row[value_header] * default_unit
            uncertainty = 0.0 * default_unit

            physical_property = property_type(
                thermodynamic_state=thermodynamic_state,
                phase=phase,
                substance=substance,
                value=value,
                uncertainty=uncertainty,
                source=source,
            )

            properties.append(physical_property)

    return_value.add_properties(*properties)
    return return_value


def data_frame_to_smiles_tuples(data_frame):
    """Extracts the smiles patterns of the components in each
    substance in a given data frame.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame to extract the component smiles from.

    Returns
    -------
    list of tuple of str
        The smiles patterns of the measured substances.
    """
    n_components = data_frame[f"N Components"].max()

    all_smiles = [
        data_frame[f"Component {i + 1}"].tolist() for i in range(n_components)
    ]

    smiles_tuples = list(zip(*all_smiles))
    smiles_tuples = list(set(tuple(sorted(x)) for x in smiles_tuples))

    return smiles_tuples
