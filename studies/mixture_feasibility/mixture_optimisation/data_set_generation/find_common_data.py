import logging
import os

import pandas
from evaluator.properties import Density, EnthalpyOfMixing, ExcessMolarVolume

from nistdataselection.curation.filtering import filter_by_substance_composition
from nistdataselection.processing import save_processed_data_set
from nistdataselection.utils.pandas import data_frame_to_smiles_tuples
from nistdataselection.utils.utils import SubstanceType, data_frame_to_pdf


def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    h_mix_data_frame = pandas.read_csv(
        "../../data_availability/all_alcohol_ester_data/enthalpy_of_mixing_binary.csv"
    )
    v_excess_data_frame = pandas.read_csv(
        "../../data_availability/converted_density_data/excess_molar_volume_binary.csv"
    )
    binary_density_data_frame = pandas.read_csv(
        "../../data_availability/converted_density_data/density_binary.csv"
    )

    h_mix_substances = set(data_frame_to_smiles_tuples(h_mix_data_frame))
    v_excess_substances = set(data_frame_to_smiles_tuples(v_excess_data_frame))
    binary_density_substances = set(
        data_frame_to_smiles_tuples(binary_density_data_frame)
    )

    # Save those data points for which we have both hmix and vexcess
    h_mix_v_excess_overlap = h_mix_substances.intersection(v_excess_substances)

    h_mix_with_v_excess = filter_by_substance_composition(
        h_mix_data_frame,
        compositions_to_include=h_mix_v_excess_overlap,
        compositions_to_exclude=None,
    )
    v_excess_with_h_mix = filter_by_substance_composition(
        v_excess_data_frame,
        compositions_to_include=h_mix_v_excess_overlap,
        compositions_to_exclude=None,
    )

    h_mix_v_excess_directory = os.path.join("common_data", "h_mix_and_v_excess")
    os.makedirs(h_mix_v_excess_directory, exist_ok=True)

    save_processed_data_set(
        h_mix_v_excess_directory,
        h_mix_with_v_excess,
        EnthalpyOfMixing,
        SubstanceType.Binary,
    )
    save_processed_data_set(
        h_mix_v_excess_directory,
        v_excess_with_h_mix,
        ExcessMolarVolume,
        SubstanceType.Binary,
    )

    data_frame_to_pdf(
        h_mix_with_v_excess, os.path.join("common_data", "h_mix_and_v_excess.pdf")
    )

    # Save those data points for which we have both hmix and binary density
    h_mix_binary_density_overlap = h_mix_substances.intersection(
        binary_density_substances
    )

    h_mix_with_binary_density = filter_by_substance_composition(
        h_mix_data_frame,
        compositions_to_include=h_mix_binary_density_overlap,
        compositions_to_exclude=None,
    )
    binary_density_with_h_mix = filter_by_substance_composition(
        binary_density_data_frame,
        compositions_to_include=h_mix_binary_density_overlap,
        compositions_to_exclude=None,
    )

    h_mix_binary_density_directory = os.path.join(
        "common_data", "h_mix_and_binary_density"
    )
    os.makedirs(h_mix_binary_density_directory, exist_ok=True)

    save_processed_data_set(
        h_mix_binary_density_directory,
        h_mix_with_binary_density,
        EnthalpyOfMixing,
        SubstanceType.Binary,
    )
    save_processed_data_set(
        h_mix_binary_density_directory,
        binary_density_with_h_mix,
        Density,
        SubstanceType.Binary,
    )

    data_frame_to_pdf(
        h_mix_with_binary_density,
        os.path.join("common_data", "h_mix_and_binary_density.pdf"),
    )


if __name__ == "__main__":
    main()
