import json
import logging
import os
import warnings

import numpy
import scipy.stats
from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection.processing import load_processed_data_set, process_raw_data
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import property_to_snake_case

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
logging.getLogger("openforcefield").setLevel(logging.ERROR)


def main():

    raw_data_directory = "../raw_archives"
    processed_data_directory = "data_with_uncertainties"

    output_directory = "uncertainties"
    os.makedirs(output_directory, exist_ok=True)

    # Convert the raw ThermoML data files into more easily manipulable
    # `pandas.DataFrame` objects.
    if not os.path.isdir(processed_data_directory):

        process_raw_data(
            directory=raw_data_directory,
            output_directory=processed_data_directory,
            retain_values=True,
            retain_uncertainties=True,
            n_processes=20,
            files_per_worker=50,
        )

    # Specify the properties to extract the modal uncertainties of.
    properties_of_interest = [
        (Density, SubstanceType.Pure),
        (Density, SubstanceType.Binary),
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (ExcessMolarVolume, SubstanceType.Binary),
        (EnthalpyOfMixing, SubstanceType.Binary),
    ]

    for property_type, substance_type in properties_of_interest:

        data_frame = load_processed_data_set(
            processed_data_directory, property_type, substance_type
        )

        if len(data_frame) == 0:
            continue

        default_unit = property_type.default_unit()
        uncertainty_header = f"{property_type.__name__} Uncertainty ({default_unit:~})"

        # Drop NaN or unbelievably high uncertainties.
        data_frame.dropna(subset=[uncertainty_header], inplace=True)
        data_frame = data_frame[data_frame[uncertainty_header] < 5.0]

        raw_uncertainties = data_frame[uncertainty_header]

        uncertainties = {
            "minimum": float(numpy.min(raw_uncertainties)),
            "maximum": float(numpy.max(raw_uncertainties)),
            "mean": float(numpy.mean(raw_uncertainties)),
            "mode": float(scipy.stats.mode(raw_uncertainties).mode),
        }

        # Save the uncertainties to a JSON file.
        property_type = property_to_snake_case(property_type)
        file_name = f"{property_type}_{str(substance_type.value)}.json"

        with open(os.path.join(output_directory, file_name), "w") as file:
            json.dump(uncertainties, file)


if __name__ == "__main__":
    main()
