"""
A utility for taking a collection of ThermoML data files, and
formatting them into a more easily manipulable format.
"""
import glob
import logging
import math
import os
import shutil

import pandas
import traceback
import uuid
from enum import Enum

from openforcefield.utils import quantity_to_string
from propertyestimator.substances import Substance
from simtk import unit

from propertyestimator.backends import QueueWorkerResources, DaskLSFBackend, DaskLocalClusterBackend
from propertyestimator.datasets import ThermoMLDataSet, PhysicalPropertyDataSet
from propertyestimator.utils import setup_timestamp_logging


class BackendType(Enum):
    Local = 'Local'
    LSF = 'LSF'


def setup_parallel_backend(backend_type=BackendType.Local,
                           number_of_workers=1,
                           lsf_queue='cpuqueue',
                           lsf_worker_commands=None):
    """Sets up the `PropertyEstimatorBackend` that will be used to distribute
    the data extraction from the xml files over multiple threads / compute nodes.

    Parameters
    ----------
    backend_type: BackendType
        The type of backend to set up.
    number_of_workers: int
        The number of workers to distribute the data extraction
        over. If the `backend_type` is set to `BackendType.Local`,
        this should be set to the number of CPU's available on your
        machine. Otherwise, this represents the number of compute
        workers which will be spun up in your LSF queueing system.
    lsf_queue: str, optional
        The queue to create the compute workers in when `backend_type`
        is set to `BackendType.LSF`.
    lsf_worker_commands: list of str
        A list of commands to run on each spun up worker (such as setting
        up the correct conda environment) when `backend_type` is set to
        `BackendType.LSF`.

    Returns
    -------
    PropertyEstimatorBackend
        The created and started backend.
    """

    calculation_backend = None

    if backend_type == BackendType.Local:
        calculation_backend = DaskLocalClusterBackend(number_of_workers=number_of_workers)

    elif backend_type == BackendType.LSF:

        queue_resources = QueueWorkerResources(number_of_threads=1,
                                               per_thread_memory_limit=8 * (unit.giga * unit.byte),
                                               wallclock_time_limit="01:30")

        if lsf_worker_commands is None:
            lsf_worker_commands = []

        calculation_backend = DaskLSFBackend(minimum_number_of_workers=0,
                                             maximum_number_of_workers=number_of_workers,
                                             resources_per_worker=queue_resources,
                                             queue_name=lsf_queue,
                                             setup_script_commands=lsf_worker_commands,
                                             adaptive_interval='1000ms')

    calculation_backend.start()

    return calculation_backend


def parse_thermoml_archives(file_paths, retain_values=False,
                            retain_uncertainties=False, directory='', **_):

    """Loads a number of ThermoML data xml files (making sure to
    catch errors raised by individual files), and concatenates
    them into a set of pandas csv files, one per property type.

    Notes
    -----
    This method is intended to be launched by a `PropertyEstimatorBackend`.

    Parameters
    ----------
    file_paths: list of str
        The file paths of the ThermoML xml files to load.
    retain_values: bool
        If False, all values for the measured properties will
        be stripped from the final data set.
    retain_uncertainties: bool
        If False, all uncertainties in measured property values will
        be stripped from the final data set.
    directory: str
        The directory to store the output csv files in.

    Returns
    -------
    dict of str and str
        The file paths to the created pandas csv files, where
        each key is the type of property stored in the csv file
        pointed to by the value.
    """

    import faulthandler

    fault_file = open(f'{str(uuid.uuid4())}.fault', 'w')
    faulthandler.enable(fault_file)

    data_set_paths = {}

    try:
        from propertyestimator.datasets import registered_thermoml_properties

        property_data_sets = {}

        for thermoml_name in registered_thermoml_properties:

            property_type = registered_thermoml_properties[thermoml_name].class_type.__name__
            property_data_sets[property_type] = PhysicalPropertyDataSet()

        # We make sure to wrap each of the 'error prone' calls in this method
        # in try-catch blocks to stop workers from being killed.
        for file_path in file_paths:

            logging.info(f'Loading ThermoML archive from: {file_path}')

            try:
                data_set = ThermoMLDataSet.from_file(file_path)

            except Exception as e:

                formatted_exception = traceback.format_exception(None, e, e.__traceback__)
                logging.info(f'An exception was raised when loading {file_path}: {formatted_exception}')

                continue

            # A data set will be none if no 'valid' properties were found
            # in the archive file.
            if data_set is None:
                continue

            for substance_id in data_set.properties:

                for physical_property in data_set.properties[substance_id]:

                    property_type = physical_property.__class__.__name__

                    if substance_id not in property_data_sets[property_type].properties:
                        property_data_sets[property_type].properties[substance_id] = []

                    property_data_sets[property_type].properties[substance_id].append(physical_property)

        unique_id = str(uuid.uuid4()).replace('-', '')

        for property_type in property_data_sets:

            file_path = os.path.join(directory, f'{property_type}_{unique_id}.csv')
            data_set_paths[property_type] = file_path

            try:
                data_set_to_csv(property_data_sets[property_type], file_path,
                                retain_values, retain_uncertainties)

            except Exception as e:

                formatted_exception = traceback.format_exception(None, e, e.__traceback__)
                logging.info(f'An exception was raised when saving the csv file of {property_type}'
                             f'properties to {file_path}: {formatted_exception}')

                continue

    except Exception as e:

        formatted_exception = traceback.format_exception(None, e, e.__traceback__)
        logging.info(f'An uncaught exception was raised: {formatted_exception}')

    faulthandler.disable()
    fault_file.close()

    return data_set_paths


def data_set_to_csv(data_set, file_path, retain_values=False, retain_uncertainties=False):
    """Saves a `propertyestimator.datasets.PhysicalPropertyDataSet`
    to a pandas csv file. It is assumed that the data set only contains
    one type of property.

    Notes
    -----
    The csv file will have columns:

        - Temperature (K)
        - Pressure (kPa)
        - Number Of Components
        - Component 1
        - Mole Fraction 1
        - Component 2
        - Mole Fraction 2
        - Component 3
        - Mole Fraction 3
        - Source

    and optionally

        - Value
        - Uncertainty

    depending on the values of `retain_values` and `retain_uncertainties`.

    The values of the component and mole fraction columns may be empty
    depending on the number of components.

    Parameters
    ----------
    data_set: propertyestimator.datasets.PhysicalPropertyDataSet
        The data set to save as a .csv file.
    file_path: str
        The file path to save the data set to.
    retain_values: bool
        If False, all values for the measured properties will
        be stripped from the final data set.
    retain_uncertainties: bool
        If False, all uncertainties in measured property values will
        be stripped from the final data set.
    """

    data_rows = []

    for substance_id in data_set.properties:

        for physical_property in data_set.properties[substance_id]:

            temperature = physical_property.thermodynamic_state.temperature.value_in_unit(unit.kelvin)
            pressure = None

            if physical_property.thermodynamic_state.pressure is not None:
                pressure = physical_property.thermodynamic_state.pressure.value_in_unit(unit.kilopascal)

            number_of_components = physical_property.substance.number_of_components

            components = [(None, None), (None, None), (None, None)]

            for index, component in enumerate(physical_property.substance.components):

                amount = physical_property.substance.get_amount(component)
                assert isinstance(amount, Substance.MoleFraction)

                components[index] = (component.smiles, amount.value)

            value = quantity_to_string(physical_property.value)
            uncertainty = quantity_to_string(physical_property.uncertainty)

            source = physical_property.source.reference

            if source is None:
                source = physical_property.source.doi

            data_rows.append({'Temperature (K)': temperature,
                              'Pressure (kPa)': pressure,
                              'Number Of Components': number_of_components,
                              'Component 1': components[0][0],
                              'Mole Fraction 1': components[0][1],
                              'Component 2': components[1][0],
                              'Mole Fraction 2': components[1][1],
                              'Component 3': components[2][0],
                              'Mole Fraction 3': components[2][1],
                              'Value': value,
                              'Uncertainty': uncertainty,
                              'Source': source})

    data_frame = pandas.DataFrame(data_rows, columns=[
        'Temperature (K)',
        'Pressure (kPa)',
        'Number Of Components',
        'Component 1',
        'Mole Fraction 1',
        'Component 2',
        'Mole Fraction 2',
        'Component 3',
        'Mole Fraction 3',
        'Value',
        'Uncertainty',
        'Source'
    ])

    if not retain_uncertainties:
        data_frame.drop(columns="Uncertainty", inplace=True)

    if not retain_values:
        data_frame.drop(columns="Value", inplace=True)

    data_frame.to_csv(file_path, index=False)


def extract_data_from_archives(archive_file_paths, compute_backend, files_per_worker=200,
                               delete_temporary_files=True):
    """Uses the compute backend to extract the data contained in
    a set of ThermoML xml data files, and then merges this data into
    convenient pandas csv files.

    Parameters
    ----------
    archive_file_paths: list of str
        The list of file paths to extract data from.
    compute_backend: PropertyEstimatorBackend
        The backend to distribute the data extraction over.
    files_per_worker: int
        The number of files to process on each compute worker
        at any one time. The file list is split into batches of
        this size to avoid memory issues on any one worker.
    delete_temporary_files: bool
        If true, all temporary files will be deleted.

    Returns
    -------
    dict of str and pandas.DataFrame
        A dictionary of pandas data frames which contain
        the unfiltered extracted data.
    """
    from propertyestimator.datasets import registered_thermoml_properties

    # Create a working directory to store temporary files in.
    working_directory_path = 'working_directory'
    os.makedirs(working_directory_path, exist_ok=True)

    # Store pointers to where the data will exist once the
    # compute backend has finished extracting it.
    calculation_futures = []

    # Submit the list of data paths to the backend in batches.
    # noinspection PyTypeChecker
    total_number_of_files = len(archive_file_paths)

    for batch_index in range(math.ceil(total_number_of_files / files_per_worker)):

        start_index = files_per_worker * batch_index
        end_index = files_per_worker * (batch_index + 1)

        if end_index >= total_number_of_files:
            end_index = total_number_of_files

        if end_index - start_index <= 0:
            continue

        worker_file_paths = archive_file_paths[start_index: end_index]

        calculation_future = compute_backend.submit_task(parse_thermoml_archives,
                                                         worker_file_paths,
                                                         False,
                                                         False,
                                                         working_directory_path)

        calculation_futures.append(calculation_future)

    extracted_data_paths = {}
    full_data_frames = {}

    for thermoml_name in registered_thermoml_properties:

        property_type = registered_thermoml_properties[thermoml_name].class_type.__name__

        extracted_data_paths[property_type] = []

        full_data_frames[property_type] = pandas.DataFrame(columns=[
            'Temperature (K)',
            'Pressure (kPa)',
            'Number Of Components',
            'Component 1',
            'Mole Fraction 1',
            'Component 2',
            'Mole Fraction 2',
            'Component 3',
            'Mole Fraction 3',
            'Value',
            'Uncertainty',
            'Source'
        ])

    while len(calculation_futures) > 0:

        current_future = calculation_futures.pop(0)
        data_paths = current_future.result()

        for property_type in data_paths:

            data_frame = pandas.read_csv(data_paths[property_type])

            full_data_frames[property_type] = pandas.concat([full_data_frames[property_type],
                                                            data_frame], ignore_index=True, sort=False)

        current_future.release()
        del current_future

    if delete_temporary_files and os.path.isdir(working_directory_path):
        shutil.rmtree(working_directory_path)

    return full_data_frames


def main():
    """Extracts all of the physical property data from a collection of
    ThermoML xml archives in a specified directory.
    """

    # Set up verbose logging.
    setup_timestamp_logging()

    # Define the directory in which to search for the xml files, and
    # find all xml file paths within that directory.
    home_directory = os.path.expanduser("~")

    archive_directory = os.path.join(home_directory, 'checked_thermoml_files')
    archive_paths = glob.glob(os.path.join(archive_directory, '*.xml'))

    # Create the backend which will distribute the extraction of data across
    # multiple threads / nodes.
    # compute_backend = setup_parallel_backend(backend_type=BackendType.Local,
    #                                          number_of_workers=4)

    # The below will optionally distribute the data extraction over nodes
    # accessible through an LSF queueing system.
    worker_script_commands = [
        f'export OE_LICENSE="{os.path.join(home_directory, "oe_license.txt")}"',
        f'. {os.path.join(home_directory, "miniconda3/etc/profile.d/conda.sh")}',
        f'conda activate nistdataselection',
    ]

    logging.info(f'Worker extra script commands: {worker_script_commands}')

    compute_backend = setup_parallel_backend(backend_type=BackendType.LSF,
                                             number_of_workers=20,
                                             lsf_worker_commands=worker_script_commands)

    # Extract the data from the archives
    data_frames = extract_data_from_archives(archive_file_paths=archive_paths,
                                             compute_backend=compute_backend)

    # Save the data frames to disk.
    output_directory = 'all_properties'
    os.makedirs(output_directory, exist_ok=True)

    for property_type in data_frames:

        data_frame = data_frames[property_type]
        data_frame.to_csv(os.path.join(output_directory, f'{property_type}.csv'))

    # Close down all of the compute workers.
    compute_backend.stop()


if __name__ == '__main__':
    main()
