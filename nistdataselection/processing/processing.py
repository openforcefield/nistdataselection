"""
A utility for taking a collection of ThermoML data files, and
formatting them into a more easily manipulable format.
"""
import glob
import logging
import math
import os
import shutil
import traceback
import uuid

import pandas
from propertyestimator.backends import DaskLocalCluster
from propertyestimator.datasets import ThermoMLDataSet, PhysicalPropertyDataSet
from propertyestimator.utils import setup_timestamp_logging

from nistdataselection.utils import PandasDataSet


def _parse_thermoml_archives(file_paths, retain_values=False,
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

            try:
                data_set = ThermoMLDataSet.from_file(file_path)

            except Exception as e:

                formatted_exception = traceback.format_exception(None, e, e.__traceback__)
                logging.warning(f'An exception was raised when loading {file_path}: {formatted_exception}')

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

                data_frame = PandasDataSet.to_pandas_data_frame(property_data_sets[property_type])

                if not retain_uncertainties and 'Uncertainty' in data_frame:
                    data_frame.drop(columns='Uncertainty', inplace=True)

                if not retain_values and 'Value' in data_frame:
                    data_frame.drop(columns='Value', inplace=True)

                data_frame.to_csv(file_path, index=False)

            except Exception as e:

                formatted_exception = traceback.format_exception(None, e, e.__traceback__)
                logging.warning(f'An exception was raised when saving the csv file of {property_type}'
                                f'properties to {file_path}: {formatted_exception}')

                continue

    except Exception as e:

        formatted_exception = traceback.format_exception(None, e, e.__traceback__)
        logging.info(f'An uncaught exception was raised: {formatted_exception}')

    return data_set_paths


def _extract_data_from_archives(archive_file_paths, compute_backend, retain_values=False,
                                retain_uncertainties=False, files_per_worker=50,
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
    retain_values: bool
        If False, all values for the measured properties will
        be stripped from the final data set.
    retain_uncertainties: bool
        If False, all uncertainties in measured property values will
        be stripped from the final data set.
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

        calculation_future = compute_backend.submit_task(_parse_thermoml_archives,
                                                         worker_file_paths,
                                                         retain_values,
                                                         retain_uncertainties,
                                                         working_directory_path)

        calculation_futures.append(calculation_future)

    extracted_data_paths = {}
    full_data_frames = {}

    for thermoml_name in registered_thermoml_properties:

        property_type = registered_thermoml_properties[thermoml_name].class_type.__name__

        extracted_data_paths[property_type] = []
        full_data_frames[property_type] = None

    total_futures = len(calculation_futures)

    while len(calculation_futures) > 0:

        current_future = calculation_futures.pop(0)
        data_paths = current_future.result()

        for property_type in data_paths:

            data_frame = pandas.read_csv(data_paths[property_type])

            if full_data_frames[property_type] is None:

                full_data_frames[property_type] = data_frame
                continue

            full_data_frames[property_type] = pandas.concat([full_data_frames[property_type],
                                                            data_frame], ignore_index=True, sort=False)

        current_future.release()
        del current_future

        logging.info(f'Finished processing {total_futures - len(calculation_futures)} '
                     f'out of {total_futures} batches (each of size {files_per_worker})')

    if delete_temporary_files and os.path.isdir(working_directory_path):
        shutil.rmtree(working_directory_path)

    return full_data_frames


def process_raw_data(directory, output_directory='property_data', retain_values=False,
                     retain_uncertainties=False, compute_backend=None, files_per_worker=50):
    """Extracts all of the physical property data from a collection of
    ThermoML xml archives in a specified directory, and converts them into
    more manageable `pandas.DataFrame` compatible csv files.

    Parameters
    ----------
    directory: str
        The directory which contains the ThermoML .xml archive files.
    output_directory: str
        The path to a directory in which to store the extracted data
        files.
    retain_values: bool
        If False, all values for the measured properties will
        be stripped from the final data set.
    retain_uncertainties: bool
        If False, all uncertainties in measured property values will
        be stripped from the final data set.
    compute_backend: PropertyEstimatorBackend, optional
        The compute backend to distribute the processing over. This is
        useful when processing a large number of archive files in one go.
        If None, a single worker `DaskLocalCluster` will be used.
    files_per_worker: int
        The number of files each worker should process at once. This
        should be lowered if segmentation faults ( / core dumps) are
        observed.
    """

    # Set up verbose logging.
    setup_timestamp_logging()

    # Define the directory in which to search for the xml files, and
    # find all xml file paths within that directory.
    archive_paths = glob.glob(os.path.join(directory, '*.xml'))

    # Create the backend which will distribute the extraction of data across
    # multiple threads / nodes.
    if compute_backend is None:
        compute_backend = DaskLocalCluster(number_of_workers=1)
        compute_backend.start()

    # Extract the data from the archives
    data_frames = _extract_data_from_archives(archive_file_paths=archive_paths,
                                              retain_values=retain_values,
                                              retain_uncertainties=retain_uncertainties,
                                              compute_backend=compute_backend,
                                              files_per_worker=files_per_worker)

    # Save the data frames to disk.
    os.makedirs(output_directory, exist_ok=True)

    for property_type in data_frames:

        data_frame = data_frames[property_type]

        # Save one file for each composition type.
        for index, data_type in enumerate(['pure', 'binary', 'ternary']):

            data_subset = data_frame.loc[data_frame['Number Of Components'] == index + 1]
            data_subset.to_csv(os.path.join(output_directory, f'{property_type}_{data_type}.csv'))
