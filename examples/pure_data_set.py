import logging
import os

from evaluator import unit
from evaluator.properties import Density, EnthalpyOfVaporization
from pkg_resources import resource_filename

from nistdataselection import processing, reporting
from nistdataselection.curation import filtering, selection
from nistdataselection.utils.utils import SubstanceType


def main():
    """Collates a directory of NIST ThermoML archive files into
    more readily manipulable pandas csv files.
    """

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    raw_data_directory = resource_filename(
        "nistdataselection", os.path.join("data", "thermoml")
    )
    processed_data_directory = "processed_data"

    # Convert the raw ThermoML data files into more easily manipulable
    # `pandas.DataFrame` objects.
    processing.process_raw_data(
        directory=raw_data_directory,
        output_directory=processed_data_directory,
        retain_values=True,
        retain_uncertainties=True,
    )

    # Define the ranges of temperatures and pressures of interest.
    # Here we choose a range of temperatures which are biologically
    # relevant (15 C - 45 C) and pressures which are close to ambient.
    temperature_range = (288.15 * unit.kelvin, 323.15 * unit.kelvin)
    pressure_range = (0.95 * unit.atmosphere, 1.05 * unit.atmosphere)

    # Define the elements that we are interested in. Here we only allow
    # a subset of those elements for which Parsley has parameters for,
    # and for which there exists plentiful data in the ThermoML archives.
    allowed_elements = ["H", "N", "C", "O", "S", "F", "Cl", "Br", "I"]

    # Define the target number of unique substances to choose for each
    # type of property of interest.
    target_substances_per_property = {
        (Density, SubstanceType.Pure): 1,
        (EnthalpyOfVaporization, SubstanceType.Pure): 1,
    }

    # Create a directory to store the filtered data in.
    filtered_data_directory = "filtered_data"
    os.makedirs(filtered_data_directory, exist_ok=True)

    # Perform basic filtering on the data sets.
    for property_type, substance_type in target_substances_per_property:

        # Load the full data sets from the processed data file
        logging.info(
            f"Applying filters to the {substance_type.value} "
            f"{property_type.__name__} data set."
        )
        data_set = processing.load_processed_data_set(
            processed_data_directory, property_type, substance_type
        )

        # Apply a standard set of filters.
        data_set = filtering.apply_standard_filters(
            data_set, temperature_range, pressure_range, allowed_elements
        )

        logging.info(f"The filtered data set contains {len(data_set)} " f"properties.")

        # Save the filtered data set.
        processing.save_processed_data_set(
            filtered_data_directory, data_set, property_type, substance_type
        )

    # Choose a set of unique substances to train the VdW parameters against.
    # These are just tuples of smiles patterns which define the composition of
    # the substance. We choose the actual mole fractions of components in a later
    # step.
    #
    # Here we specify which regions of chemical space we want to cover. This
    # is mainly driven by the VdW parameters we wish to exercise, but may also
    # be supplemented with additional environments which are poorly represented.
    target_environments = [
        "[#1:1]-[#6X4]",
        "[#1:1]-[#6X3]",
        "[#1:1]-[#8]",
        "[#6:1]",
        "[#6X4:1]",
        "[#8:1]",
        "[#8X2H0+0:1]",
        "[#8X2H1+0:1]",
        "[#7:1]",
        "[#16:1]",
        "[#9:1]",
        "[#17:1]",
        "[#35:1]",
    ]

    chosen_substances = selection.select_substances(
        filtered_data_directory, target_substances_per_property, target_environments
    )

    logging.info(f"{len(chosen_substances)} substances where chosen.")

    # Define the specific states at which we wish to select data. These are currently
    # tuples of temperature, pressure, and a tuple of the mole fractions of each of the
    # components.
    density_target_state_points = [
        selection.StatePoint(298.15 * unit.kelvin, 101.325 * unit.kilopascal, (1.0,)),
        selection.StatePoint(318.15 * unit.kelvin, 101.325 * unit.kilopascal, (1.0,)),
    ]
    hvap_target_state_points = [
        selection.StatePoint(298.15 * unit.kelvin, 101.325 * unit.kilopascal, (1.0,)),
        selection.StatePoint(318.15 * unit.kelvin, 101.325 * unit.kilopascal, (1.0,)),
    ]

    target_property_state_points = {
        (Density, SubstanceType.Pure): density_target_state_points,
        (EnthalpyOfVaporization, SubstanceType.Pure): hvap_target_state_points,
    }

    # Set the output path to the data set.
    data_set_name = "pure_data_set"

    # Choose the final data set containing the chosen substances, and
    # data points at the target state points.
    data_set = selection.select_data_points(
        filtered_data_directory, chosen_substances, target_property_state_points
    )

    with open(f"{data_set_name}.json", "w") as file:
        file.write(data_set.json())

    data_set.to_pandas().to_csv(f"{data_set_name}.csv")

    # Generate a pdf report detailing the chosen set.
    reporting.generate_report(
        f"{data_set_name}.json", vdw_smirks_of_interest=target_environments
    )


if __name__ == "__main__":
    main()
