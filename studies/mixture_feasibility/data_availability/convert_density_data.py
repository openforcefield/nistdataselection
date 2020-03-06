"""The purpose of this script is to (where possible) convert any binary
mass density data in excess molar volume data and conversely, all
excess molar volume data into binary mass data.

This is only possible where pure density data is available for the
substances which the binary properties where measured for, and at
the same temperatures and pressures.
"""
import logging
import os

import pandas
from evaluator import unit
from evaluator.properties import Density, ExcessMolarVolume

from nistdataselection.curation.filtering import filter_duplicates
from nistdataselection.processing import (
    load_processed_data_set,
    save_processed_data_set,
)
from nistdataselection.utils import SubstanceType, get_molecular_weight
from nistdataselection.utils.utils import data_frame_to_pdf

logger = logging.getLogger(__name__)


def find_overlapping_data_points(pure_data_set, binary_data_set):
    """Finds those binary data points for which there also exists pure
     data points for each component in the binary system.

    Parameters
    ----------
    pure_data_set: pandas.DataFrame
        The pure data set.
    binary_data_set: pandas.DataFrame
        The binary data set.

    Returns
    -------
    pandas.DataFrame
        The data set containing the pure and binary data points
        measured for the same substances at the same state pounts
    """

    pure_data_set.dropna(axis=1, how="all", inplace=True)
    binary_data_set.dropna(axis=1, how="all", inplace=True)

    pure_data_set["Temperature (K)"] = pure_data_set["Temperature (K)"].round(2)
    pure_data_set["Pressure (kPa)"] = pure_data_set["Pressure (kPa)"].round(1)

    binary_data_set["Temperature (K)"] = binary_data_set["Temperature (K)"].round(2)
    binary_data_set["Pressure (kPa)"] = binary_data_set["Pressure (kPa)"].round(1)

    pure_data_set = pandas.merge(
        pure_data_set,
        pure_data_set,
        how="inner",
        on=["Temperature (K)", "Pressure (kPa)"],
    )

    overlapping_set = pandas.merge(
        binary_data_set,
        pure_data_set,
        how="inner",
        left_on=["Temperature (K)", "Pressure (kPa)", "Component 1", "Component 2"],
        right_on=[
            "Temperature (K)",
            "Pressure (kPa)",
            "Component 1_x",
            "Component 1_y",
        ],
    )

    return overlapping_set


def convert_density_to_v_excess(density_data_set):
    """Converts a pandas data frame containing both binary mass densities
    and pure mass densities into one which contains excess molar volume
    measurements.

    Parameters
    ----------
    density_data_set: pandas.DataFrame
        The data frame containing both pure and binary
        density measurements. This should be generated using the
        `find_overlapping_data_points` function.

    Returns
    -------
    pandas.DataFrame
        A data frame which contains the excess molar volume measurements.
    """

    def molecular_weight(smiles):
        return get_molecular_weight(smiles).to(unit.gram / unit.mole).magnitude

    m_1 = density_data_set["Component 1"].apply(molecular_weight)
    m_1_x_1 = m_1 * density_data_set["Mole Fraction 1"]

    m_2 = density_data_set["Component 2"].apply(molecular_weight)
    m_2_x_2 = m_2 * density_data_set["Mole Fraction 2"]

    v_excess = (
        (m_1_x_1 + m_2_x_2) / density_data_set["Density Value (g / ml)"]
        - m_1_x_1 / density_data_set["Density Value (g / ml)_x"]
        - m_2_x_2 / density_data_set["Density Value (g / ml)_y"]
    )

    source = density_data_set[["Source", "Source_x", "Source_y"]].agg(
        " + ".join, axis=1
    )

    v_excess_data_set = density_data_set[
        [
            "Temperature (K)",
            "Pressure (kPa)",
            "Phase",
            "N Components",
            "Component 1",
            "Role 1",
            "Mole Fraction 1",
            "Component 2",
            "Role 2",
            "Mole Fraction 2",
        ]
    ]

    v_excess_data_set.insert(
        v_excess_data_set.shape[1], "ExcessMolarVolume Value (cm ** 3 / mol)", v_excess
    )
    v_excess_data_set.insert(v_excess_data_set.shape[1], "Source", source)

    return v_excess_data_set


def convert_v_excess_to_density(v_excess_data_set):
    """Converts a pandas data frame containing both excess molar volumes
    and pure mass densities into one which contains binary mass density
    measurements.

    Parameters
    ----------
    v_excess_data_set: pandas.DataFrame
        The data frame containing both pure density and excess molar
        volume measurements. This should be generated using the
        `find_overlapping_data_points` function.

    Returns
    -------
    pandas.DataFrame
        A data frame which contains the excess molar volume measurements.
    """

    def molecular_weight(smiles):
        return get_molecular_weight(smiles).to(unit.gram / unit.mole).magnitude

    m_1 = v_excess_data_set["Component 1"].apply(molecular_weight)
    m_1_x_1 = m_1 * v_excess_data_set["Mole Fraction 1"]

    m_2 = v_excess_data_set["Component 2"].apply(molecular_weight)
    m_2_x_2 = m_2 * v_excess_data_set["Mole Fraction 2"]

    v_excess = v_excess_data_set["ExcessMolarVolume Value (cm ** 3 / mol)"]

    denominator = (
        v_excess
        + m_1_x_1 / v_excess_data_set["Density Value (g / ml)_x"]
        + m_2_x_2 / v_excess_data_set["Density Value (g / ml)_y"]
    )

    rho_binary = (m_1_x_1 + m_2_x_2) / denominator

    source = v_excess_data_set[["Source", "Source_x", "Source_y"]].agg(
        " + ".join, axis=1
    )

    density_data_set = v_excess_data_set[
        [
            "Temperature (K)",
            "Pressure (kPa)",
            "Phase",
            "N Components",
            "Component 1",
            "Role 1",
            "Mole Fraction 1",
            "Component 2",
            "Role 2",
            "Mole Fraction 2",
        ]
    ]

    density_data_set.insert(
        density_data_set.shape[1] - 1, "Density Value (g / ml)", rho_binary
    )
    density_data_set.insert(density_data_set.shape[1] - 1, "Source", source)

    return density_data_set


def main():

    output_directory = "converted_density_data"
    os.makedirs(output_directory, exist_ok=True)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    root_data_directory = os.path.join("..", "..", "shared", "filtered_data")

    # Load in the data sets of interest
    pure_density_data = load_processed_data_set(
        root_data_directory, Density, SubstanceType.Pure
    )
    binary_density_data = load_processed_data_set(
        root_data_directory, Density, SubstanceType.Binary
    )
    v_excess_data = load_processed_data_set(
        root_data_directory, ExcessMolarVolume, SubstanceType.Binary
    )

    # Add the pure data to the binary data sets to make conversion easier (TM).
    combined_binary_density_data = find_overlapping_data_points(
        pure_density_data, binary_density_data
    )
    combined_v_excess_data = find_overlapping_data_points(
        pure_density_data, v_excess_data
    )

    # Inter-convert the two sets
    v_excess_from_density = convert_density_to_v_excess(combined_binary_density_data)
    density_from_v_excess = convert_v_excess_to_density(combined_v_excess_data)

    # Add the converted data to the full data.
    full_binary_data = pandas.concat(
        [binary_density_data, density_from_v_excess], ignore_index=True, sort=False
    )
    full_v_excess_data = pandas.concat(
        [v_excess_data, v_excess_from_density], ignore_index=True, sort=False
    )

    # Filter out any duplicate data points.
    full_binary_data = filter_duplicates(full_binary_data)
    full_v_excess_data = filter_duplicates(full_v_excess_data)

    save_processed_data_set(
        output_directory, full_binary_data, Density, SubstanceType.Binary
    )
    save_processed_data_set(
        output_directory, full_v_excess_data, ExcessMolarVolume, SubstanceType.Binary
    )

    # Save the converted data sets out as pdf files.
    data_frame_to_pdf(
        density_from_v_excess,
        os.path.join(output_directory, "density_from_v_excess.pdf"),
    )
    data_frame_to_pdf(
        v_excess_from_density,
        os.path.join(output_directory, "v_excess_from_density.pdf"),
    )


if __name__ == "__main__":
    main()
