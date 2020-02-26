import logging
import warnings

from evaluator import unit
from evaluator.properties import EnthalpyOfVaporization

from nistdataselection.processing import (
    load_processed_data_set,
    process_raw_data,
    save_processed_data_set,
)
from nistdataselection.utils import SubstanceType

warnings.filterwarnings("ignore")
logging.getLogger("openforcefield").setLevel(logging.ERROR)


def main():

    raw_data_directory = "raw_archives"
    processed_data_directory = "processed_data"

    # Convert the raw ThermoML data files into more easily manipulable
    # `pandas.DataFrame` objects.
    process_raw_data(
        directory=raw_data_directory,
        output_directory=processed_data_directory,
        retain_values=True,
        retain_uncertainties=False,
        n_processes=20,
        files_per_worker=50,
    )

    # Here we will also 'fix' the enthalpy of vaporization entries so that
    # they have a pressure (approximated as ambient).
    h_vap_properties = [
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Binary),
        (EnthalpyOfVaporization, SubstanceType.Ternary),
    ]

    pressure = 1.0 * unit.atmosphere

    for property_tuple in h_vap_properties:

        data_set = load_processed_data_set(processed_data_directory, *property_tuple)

        data_set["Pressure (kPa)"] = data_set["Pressure (kPa)"].fillna(
            pressure.to(unit.kilopascal).magnitude
        )

        save_processed_data_set(processed_data_directory, data_set, *property_tuple)


if __name__ == "__main__":
    main()
