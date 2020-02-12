"""
A utility for going between `evaluator.datasets.PhysicalPropertyDataSet
objects and `pandas.DataFrame` objects.
"""

import evaluator.properties
import numpy
import pandas
from evaluator import unit
from evaluator.attributes import UNDEFINED
from evaluator.datasets import MeasurementSource, PhysicalPropertyDataSet, PropertyPhase
from evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from evaluator.thermodynamics import ThermodynamicState


class PandasDataSet(PhysicalPropertyDataSet):
    """A helper class for importing and physical property
    data sets from pandas `DataFrame` objects and csv files.
    """

    @classmethod
    def from_pandas(cls, data_frame):
        """Converts a `pandas.DataFrame` to a `PhysicalPropertyDataSet` object.
        See the `to_pandas()` function for information on the required columns.

        Parameters
        ----------
        data_frame: pandas.DataFrame
            The data frame to convert.

        Returns
        -------
        PhysicalPropertyDataSet
            The converted data set.
        """

        return_value = cls()

        if len(data_frame) == 0:
            return return_value

        # Make sure the base columns are present.
        required_base_columns = [
            "Temperature",
            "Pressure",
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

            assert len(column_name_split) == 2
            assert f"{column_name_split[0]} Uncertainty" in data_frame

            property_types.append(column_name_split[0])

        assert all(hasattr(evaluator.properties, x) for x in property_types)
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
            pressure = unit.Quantity(row["Pressure"])
            pressure.ito(unit.kilopascal)

            temperature = unit.Quantity(row["Temperature"])
            temperature.ito(unit.kelvin)

            thermodynamic_state = ThermodynamicState(temperature, pressure)

            phase = PropertyPhase[row["Phase"]]

            source = MeasurementSource(reference=row["Source"])

            for property_type in property_types:

                if not isinstance(row[f"{property_type} Value"], str) and numpy.isnan(
                    row[f"{property_type} Value"]
                ):
                    continue

                value = unit.Quantity(row[f"{property_type} Value"])
                uncertainty = (
                    UNDEFINED
                    if numpy.isnan(row[f"{property_type} Uncertainty"])
                    else unit.Quantity(row[f"{property_type} Uncertainty"])
                )

                property_class = getattr(evaluator.properties, property_type)

                physical_property = property_class(
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

    @classmethod
    def from_csv(cls, file_path):
        """Converts a pandas csv file to a `PhysicalPropertyDataSet` object.
        See the `to_pandas()` function for information on the required columns.

        Parameters
        ----------
        file_path: str
            The path to the csv data file.

        Returns
        -------
        PhysicalPropertyDataSet
            The converted data set.
        """
        data_frame = pandas.read_csv(file_path)
        return cls.from_pandas(data_frame)

    def to_csv(self, file_path):
        """Exports a `PhysicalPropertyDataSet` to a pandas csv file. This is convience
        method which calls `.to_pandas()` and then `to_csv()` on that data frame.

        Parameters
        ----------
        file_path: str
            The path to export the file to.
        """
        data_frame = self.to_pandas()
        data_frame.to_csv(file_path, index=False)
