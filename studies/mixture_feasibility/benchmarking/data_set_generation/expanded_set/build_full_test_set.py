"""A script which combines the pure and mixture test sets
into a single data set object.
"""
import logging
import os

import pandas

from nistdataselection.utils import data_set_from_data_frame
from nistdataselection.utils.utils import data_frame_to_pdf

logger = logging.getLogger(__name__)


def main():

    logging.basicConfig(level=logging.INFO)

    data_frames = [
        pandas.read_csv(os.path.join("test_sets", "density_binary.csv")),
        pandas.read_csv(os.path.join("test_sets", "enthalpy_of_mixing_binary.csv")),
        pandas.read_csv(os.path.join("test_sets", "excess_molar_volume_binary.csv")),
        pandas.read_csv(os.path.join("test_sets", "pure_set.csv")),
    ]

    full_data_frame = pandas.concat(data_frames, ignore_index=True, sort=False)
    full_data_set = data_set_from_data_frame(full_data_frame)

    full_data_frame.to_csv(os.path.join("test_sets", "full_set.csv"), index=False)
    full_data_set.json(os.path.join("test_sets", "full_set.json"))

    data_frame_to_pdf(full_data_frame, os.path.join("test_sets", "full_set.pdf"))


if __name__ == "__main__":
    main()
