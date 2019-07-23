"""
A utility for going between `propertyestimator.datasets.PhysicalPropertyDataSet
objects and `pandas.DataFrame` objects.
"""

import pandas
from openforcefield.utils import quantity_to_string, string_to_quantity
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.properties import PropertyPhase, PhysicalProperty, MeasurementSource
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from simtk import unit


class PandasDataSet(PhysicalPropertyDataSet):
    """A helper class for importing and exporting physical property
    data sets to pandas `DataFrame` objects and csv files.

    Notes
    -----
    This class will only work with data sets which contain a single
    type of property (e.g. only densities).
    """

    @classmethod
    def from_data_frame(cls, data_frame, property_type):
        """Converts a `pandas.DataFrame` to a `PhysicalPropertyDataSet` object.
        The data frame should only contain values for a single type of property,
        and should have columns of

        'Temperature (K)', 'Pressure (kPa)', 'Phase', 'Number Of Components', 'Component 1',
        'Mole Fraction 1', ..., 'Component N', 'Mole Fraction N', 'Value' (optional),
        'Uncertainty' (optional), 'Source'

        where 'Component X' is a column containing the smiles representation of component X.

        Parameters
        ----------
        data_frame: pandas.DataFrame
            The data frame to convert.
        property_type: class
            The type of property stored in the data set.

        Returns
        -------
        PhysicalPropertyDataSet
            The converted data set.

        Examples
        --------
        To convert a data frame containing densities

        >>> from propertyestimator.properties import Density
        >>> data_set = PandasDataSet.from_data_frame(data_frame, Density)
        """

        assert issubclass(property_type, PhysicalProperty)
        assert property_type != PhysicalProperty

        return_value = cls()

        sources = set()

        for index, row in data_frame.iterrows():

            # Create the substance from the component columns
            number_of_components = row['Number Of Components']

            substance = Substance()

            for component_index in range(number_of_components):

                smiles = row[f'Component {component_index + 1}']
                mole_fraction = row[f'Mole Fraction {component_index + 1}']

                substance.add_component(Substance.Component(smiles=smiles),
                                        Substance.MoleFraction(value=mole_fraction))

            if substance.identifier not in return_value._properties:
                return_value.properties[substance.identifier] = []

            # Parse the state
            thermodynamic_state = ThermodynamicState(temperature=row['Temperature (K)'] * unit.kelvin,
                                                     pressure=row['Pressure (kPa)'] * unit.kilopascal)

            phase = PropertyPhase(row['Phase'])

            value = None if 'Value' not in row else string_to_quantity(row['Value'])
            uncertainty = None if 'Uncertainty' not in row else string_to_quantity(row['Uncertainty'])

            source = MeasurementSource(reference=row['Source'])
            sources.add(source)

            physical_property = property_type(thermodynamic_state=thermodynamic_state,
                                              phase=phase,
                                              substance=substance,
                                              value=value,
                                              uncertainty=uncertainty,
                                              source=source)

            return_value.properties[substance.identifier].append(physical_property)

        return_value._sources = list(sources)

        return return_value

    @classmethod
    def from_pandas_csv(cls, file_path, property_type):
        """Converts a pandas csv file to a `PhysicalPropertyDataSet` object.
        The data file should only contain values for a single type of property,
        and should have columns of

        'Temperature (K)', 'Pressure (kPa)', 'Phase', 'Number Of Components', 'Component 1',
        'Mole Fraction 1', ..., 'Component N', 'Mole Fraction N', 'Value' (optional),
        'Uncertainty' (optional), 'Source'

        where 'Component X' is a column containing the smiles representation of component X.

        Parameters
        ----------
        file_path: str
            The path to the csv data file.
        property_type: class
            The type of property stored in the data file.

        Returns
        -------
        PhysicalPropertyDataSet
            The converted data set.
        """
        data_frame = pandas.read_csv(file_path)
        return cls.from_data_frame(data_frame, property_type)

    @staticmethod
    def to_pandas_data_frame(data_set):
        """Converts a `PhysicalPropertyDataSet` to a `pandas.DataFrame` object
        with columns of

        'Temperature (K)', 'Pressure (kPa)', 'Phase', 'Number Of Components', 'Component 1',
        'Mole Fraction 1', ..., 'Component N', 'Mole Fraction N', 'Value', 'Uncertainty', 'Source'

        where 'Component X' is a column containing the smiles representation of component X.

        Parameters
        ----------
        data_set: PhysicalPropertyDataSet
            The data set to convert.

        Returns
        -------
        pandas.DataFrame
            The create data frame.
        """
        data_rows = []
        property_type = None

        # Determine the maximum number of components for any
        # given measurements.
        maximum_number_of_components = 0

        for substance_id in data_set.properties:

            if len(data_set.properties[substance_id]) == 0:
                continue

            maximum_number_of_components = max(maximum_number_of_components,
                                               data_set.properties[substance_id][0].substance.number_of_components)

        # Make sure the maximum number of components is not zero.
        if maximum_number_of_components <= 0 and len(data_set.properties) > 0:

            raise ValueError('The data set did not contain any substances with '
                             'one or more components.')

        # Extract the data from the data set.
        for substance_id in data_set.properties:

            for physical_property in data_set.properties[substance_id]:

                if property_type is None:
                    property_type = type(physical_property)

                if property_type != type(physical_property):

                    raise ValueError('Only data sets containing a single type of '
                                     'property can be converted to a DataFrame '
                                     'object')

                # Extract the measured state.
                temperature = physical_property.thermodynamic_state.temperature.value_in_unit(unit.kelvin)
                pressure = None

                if physical_property.thermodynamic_state.pressure is not None:
                    pressure = physical_property.thermodynamic_state.pressure.value_in_unit(unit.kilopascal)

                phase = physical_property.phase

                # Extract the component data.
                number_of_components = physical_property.substance.number_of_components

                components = [(None, None)] * maximum_number_of_components

                for index, component in enumerate(physical_property.substance.components):

                    amount = physical_property.substance.get_amount(component)
                    assert isinstance(amount, Substance.MoleFraction)

                    components[index] = (component.smiles, amount.value)

                # Extract the value data as a string.
                # noinspection PyTypeChecker
                value = (None if physical_property.value is None else
                         quantity_to_string(physical_property.value))

                # noinspection PyTypeChecker
                uncertainty = (None if physical_property.uncertainty is None else
                               quantity_to_string(physical_property.uncertainty))

                # Extract the data source.
                source = physical_property.source.reference

                if source is None:
                    source = physical_property.source.doi

                # Create the data row.
                data_row = {
                    'Temperature (K)': temperature,
                    'Pressure (kPa)': pressure,
                    'Phase': phase,
                    'Number Of Components': number_of_components
                }

                for index in range(len(components)):
                    data_row[f'Component {index + 1}'] = components[index][0]
                    data_row[f'Mole Fraction {index + 1}'] = components[index][1]

                data_row['Value'] = value
                data_row['Uncertainty'] = uncertainty
                data_row['Source'] = source

                data_rows.append(data_row)

        # Set up the column headers.
        data_columns = [
            'Temperature (K)',
            'Pressure (kPa)',
            'Phase',
            'Number Of Components',
        ]

        for index in range(maximum_number_of_components):
            data_columns.append(f'Component {index + 1}')
            data_columns.append(f'Mole Fraction {index + 1}')

        data_columns.extend([
            'Value',
            'Uncertainty',
            'Source'
        ])

        data_frame = pandas.DataFrame(data_rows, columns=data_columns)
        return data_frame

    @staticmethod
    def to_pandas_csv(data_set, file_path):
        """Exports a `PhysicalPropertyDataSet` to a pandas csv file
        with columns of

        'Temperature (K)', 'Pressure (kPa)', 'Phase', 'Number Of Components', 'Component 1',
        'Mole Fraction 1', ..., 'Component N', 'Mole Fraction N', 'Value', 'Uncertainty', 'Source'

        where 'Component X' is a column containing the smiles representation of component X.

        Parameters
        ----------
        data_set: PhysicalPropertyDataSet
            The data set to export.
        file_path: str
            The path to export the file to.
        """
        data_frame = PandasDataSet.to_pandas_data_frame(data_set)
        data_frame.to_csv(file_path, index=False)
