"""
A utility for taking a collection of ThermoML data files, and
formatting them into a more easily manipulable format.
"""
import copy
import functools
import glob
import logging
import math
import os
from collections import defaultdict
from multiprocessing.pool import Pool

import tqdm
from evaluator.attributes import UNDEFINED
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.datasets.thermoml import ThermoMLDataSet

from nistdataselection.utils.utils import SubstanceType, substance_type_to_int

logger = logging.getLogger(__name__)


def _parse_thermoml_archives(file_paths, retain_values, retain_uncertainties, **_):
    """Loads a number of ThermoML data xml files (making sure to
    catch errors raised by individual files), and concatenates
    them into data sets containing a single type of property.

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

    Returns
    -------
    dict of str and PhysicalPropertyDataSet
        The parsed data frames.
    """

    data_sets = defaultdict(PhysicalPropertyDataSet)

    try:

        # We make sure to wrap each of the 'error prone' calls in this method
        # in try-catch blocks to stop workers from being killed.
        for file_path in file_paths:

            try:
                data_set = ThermoMLDataSet.from_file(file_path)

            except Exception:

                logger.exception(f"An exception was raised when loading {file_path}")
                continue

            # A data set will be none if no 'valid' properties were found
            # in the archive file.
            if data_set is None:
                continue

            for physical_property in data_set:

                if not retain_values:
                    physical_property.value = UNDEFINED
                if not retain_uncertainties:
                    physical_property.uncertainty = UNDEFINED

                property_type = physical_property.__class__.__name__
                data_sets[property_type].add_properties(physical_property)

    except Exception:
        logger.exception(f"An uncaught exception was raised.")
        data_sets = {}

    return data_sets


def _extract_data_from_archives(
    archive_file_paths,
    n_processes,
    retain_values,
    retain_uncertainties,
    files_per_worker=50,
):
    """Uses the compute backend to extract the data contained in
    a set of ThermoML xml data files, and then merges this data into
    convenient pandas csv files.

    Parameters
    ----------
    archive_file_paths: list of str
        The list of file paths to extract data from.
    n_processes: int
        The number of processes to use.
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

    Returns
    -------
    dict of str and PhysicalPropertyDataSet
        A dictionary of data sets which contain the extracted data.
    """

    # Submit the list of data paths to the backend in batches.
    # noinspection PyTypeChecker
    total_number_of_files = len(archive_file_paths)
    file_batches = []

    for batch_index in range(math.ceil(total_number_of_files / files_per_worker)):

        start_index = files_per_worker * batch_index
        end_index = files_per_worker * (batch_index + 1)

        if end_index >= total_number_of_files:
            end_index = total_number_of_files

        if end_index - start_index <= 0:
            continue

        worker_file_paths = archive_file_paths[start_index:end_index]
        file_batches.append(worker_file_paths)

    with Pool(n_processes) as pool:

        all_data_sets = list(
            tqdm.tqdm(
                pool.imap(
                    functools.partial(
                        _parse_thermoml_archives,
                        retain_values=retain_values,
                        retain_uncertainties=retain_uncertainties,
                    ),
                    file_batches,
                ),
                total=len(file_batches),
            )
        )

    logger.info("Combining data sets")
    data_sets = {}

    for data_sets_per_type in all_data_sets:

        for property_type, data_set in data_sets_per_type.items():

            if property_type not in data_sets:

                data_sets[property_type] = data_set
                continue

            data_sets[property_type].merge(data_set)

    return data_sets


def process_raw_data(
    directory,
    output_directory="property_data",
    retain_values=True,
    retain_uncertainties=True,
    n_processes=1,
    files_per_worker=50,
):
    """Extracts all of the physical property data from a collection of
    ThermoML xml archives in a specified directory, and converts them into
    `pandas.DataFrame` objects.

    Parameters
    ----------
    directory: str
        The directory which contains the ThermoML .xml archive files.
    output_directory: str
        The path to a directory in which to store the extracted data
        frames.
    retain_values: bool
        If False, all values for the measured properties will
        be stripped from the final data set.
    retain_uncertainties: bool
        If False, all uncertainties in measured property values will
        be stripped from the final data set.
    n_processes: int
        The number of processes to use when extracting the data.
    files_per_worker: int
        The number of files each worker should process at once. This
        should be lowered if segmentation faults ( / core dumps) are
        observed.
    """

    # Define the directory in which to search for the xml files, and
    # find all xml file paths within that directory.
    archive_paths = glob.glob(os.path.join(directory, "*.xml"))

    # Extract the data from the archives
    data_sets = _extract_data_from_archives(
        archive_file_paths=archive_paths,
        retain_values=retain_values,
        retain_uncertainties=retain_uncertainties,
        n_processes=n_processes,
        files_per_worker=files_per_worker,
    )

    # Save the data frames to disk.
    os.makedirs(output_directory, exist_ok=True)

    for property_type, data_set in data_sets.items():

        data_set.json(f"{property_type}.json")

        # Save one file for each composition type.
        for substance_type in [
            SubstanceType.Pure,
            SubstanceType.Binary,
            SubstanceType.Ternary,
        ]:

            number_of_components = substance_type_to_int[substance_type]

            substance_data_set = PhysicalPropertyDataSet.from_json(f"{property_type}.json")
            substance_data_set.filter_by_components(number_of_components)

            save_processed_data_set(
                output_directory, substance_data_set, property_type, substance_type
            )


def save_processed_data_set(directory, data_set, property_type, substance_type):
    """Saves a data set of measured physical properties of a specific
    type which was created using the `process_raw_data` function, with
    a file name of `PropertyType_SubstanceType.json`.

    Parameters
    ----------
    directory: str
        The path to the directory to save the data set in.
    data_set: PhysicalPropertyDataSet
        The data set to save.
    property_type: type of PhysicalProperty or str
        The type of property in the data set.
    substance_type: SubstanceType
        The type of substances in the data set.
    """
    os.makedirs(directory, exist_ok=True)

    # Try to load in the pandas data file.
    if not isinstance(property_type, str):
        property_type = property_type.__name__

    csv_file_name = f"{property_type}_{str(substance_type.value)}.csv"
    csv_file_path = os.path.join(directory, csv_file_name)

    data_set.to_pandas().to_csv(csv_file_path, index=False)

    json_file_name = f"{property_type}_{str(substance_type.value)}.json"
    json_file_path = os.path.join(directory, json_file_name)

    data_set.json(json_file_path)


def load_processed_data_set(directory, property_type, substance_type):
    """Loads a data set of measured physical properties of a specific
    type which was created using the `process_raw_data` function.

    Parameters
    ----------
    directory: str
        The path which contains the data csv files generated
        by the `process_raw_data` method.
    property_type: type of PhysicalProperty or str
        The property of interest.
    substance_type: SubstanceType
        The substance type of interest.

    Returns
    -------
    PhysicalPropertyDataSet
        The loaded data set.
    """

    assert os.path.isdir(directory)

    if not isinstance(property_type, str):
        property_type = property_type.__name__

    # Try to load in the pandas data file.
    file_name = f"{property_type}_{str(substance_type.value)}.json"
    file_path = os.path.join(directory, file_name)

    if not os.path.isfile(file_path):

        raise ValueError(
            f"No data file could be found for "
            f"{substance_type} {property_type}s at {file_path}"
        )

    data_set = PhysicalPropertyDataSet.from_json(file_path)
    return data_set
