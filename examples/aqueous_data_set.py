import os

from pkg_resources import resource_filename
from propertyestimator.properties import ExcessMolarVolume, EnthalpyOfMixing

from nistdataselection.curation import curate_data_set
from nistdataselection.processing import process_raw_data
from nistdataselection.reporting import generate_report
from nistdataselection.utils.utils import SubstanceType


def main():
    """Collates a directory of NIST ThermoML archive files into
    more readily manipulable pandas csv files.
    """

    raw_data_directory = resource_filename("nistdataselection", os.path.join("data", "thermoml"))
    processed_data_directory = "processed_data"

    # Convert the raw ThermoML data files into more easily manipulable
    # `pandas.DataFrame` objects.
    process_raw_data(
        directory=raw_data_directory,
        output_directory=processed_data_directory,
        retain_values=True,
        retain_uncertainties=True,
    )

    # Define the desired number of unique substances which should have data points
    # for each of the properties of interest
    desired_substances_per_property = {
        (EnthalpyOfMixing, SubstanceType.Binary): 2,
        (ExcessMolarVolume, SubstanceType.Binary): 2,
    }

    # Ideally we would like to choose compounds for which both density
    # and enthalpy of vaporisation data is available. Failing that, we
    # allow the method to fall back to choosing compounds for which only
    # density, or only enthalpy of vaporisation data is available.
    property_priority_order = [
        [
            # Failing that, we pick molecules for which we only have enthalpies
            # of mixing.
            (EnthalpyOfMixing, SubstanceType.Binary)
        ],
        [
            # Finally, choose molecules for which we only have excess molar volumes.
            (ExcessMolarVolume, SubstanceType.Binary)
        ],
    ]

    data_set_path = "aqueous_data_set.json"

    curate_data_set(
        processed_data_directory,
        property_priority_order,
        desired_substances_per_property,
        required_smiles_to_include=["O"],
        output_data_set_path=data_set_path,
    )

    generate_report(data_set_path)


if __name__ == "__main__":
    main()
