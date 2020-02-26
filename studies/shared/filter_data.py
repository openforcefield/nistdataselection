import functools
import logging
import os
from glob import glob
from multiprocessing.pool import Pool

import pandas
from evaluator import unit

from nistdataselection.curation import filtering

logging.propagate = False


def filter_data(
    input_file_path,
    output_file_path,
    temperature_range,
    pressure_range,
    allowed_elements,
):
    """Filters data for each of the specified type of property
    according to a standard set of filters.

    Parameters
    ----------
    input_file_path: str
        The file path to the pandas data frame (csv) to filter.
    output_file_path: str
        The file path to store the filtered data frame (csv) to.
    temperature_range: tuple of unit.Quantity and unit.Quantity
        The minimum and maximum temperature.
    pressure_range: tuple of unit.Quantity and unit.Quantity
        The minimum and maximum pressure.
    allowed_elements: list of str
        The list of elements which must only be present
        in components.
    """

    openff_logger = logging.getLogger("openforcefield")
    openff_logger.setLevel(logging.ERROR)

    logger = logging.getLogger()
    logger.handlers = []

    logger.setLevel(logging.DEBUG)

    logger_path = os.path.splitext(output_file_path)[0] + ".log"

    logger_handler = logging.FileHandler(logger_path)
    logger.addHandler(logger_handler)

    logger.info(f"Original file: {input_file_path}")

    try:

        data_frame = pandas.read_csv(input_file_path)

        data_frame = filtering.apply_standard_filters(
            data_frame, temperature_range, pressure_range, allowed_elements
        )

        current_number_of_properties = len(data_frame)

        data_frame = filtering.filter_ionic_liquids(data_frame)

        logger.debug(
            f"{current_number_of_properties - len(data_frame)} ionic liquids were "
            f"filtered out."
        )

        data_frame.dropna(axis=1, how="all")
        data_frame.to_csv(output_file_path, index=False)

    except Exception:
        logger.exception(f"Error processing {input_file_path}")

    logger.handlers = []


def main():

    # Create a directory to store the filtered data in.
    filtered_directory = "filtered_data"
    os.makedirs(filtered_directory, exist_ok=True)

    # Define the filter criteria.
    temperature_range = (288.15 * unit.kelvin, 323.15 * unit.kelvin)
    pressure_range = (0.95 * unit.atmosphere, 1.05 * unit.atmosphere)
    allowed_elements = ["H", "N", "C", "O", "Br", "Cl", "F", "S"]

    # Parallelize the filtering
    n_processes = 20

    input_paths = glob("processed_data/*.csv")

    output_paths = [
        os.path.join(filtered_directory, os.path.basename(x)) for x in input_paths
    ]

    with Pool(n_processes) as pool:

        pool.starmap(
            functools.partial(
                filter_data,
                temperature_range=temperature_range,
                pressure_range=pressure_range,
                allowed_elements=allowed_elements,
            ),
            zip(input_paths, output_paths),
        )


if __name__ == "__main__":
    main()
