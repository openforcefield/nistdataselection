"""The scripts will produce a pandas `DataFrame` which contains information
about which combinations of environments have which data types.
"""
import itertools
import logging
import os
from glob import glob

import pandas
from evaluator.properties import Density, EnthalpyOfMixing

from nistdataselection.processing import load_processed_data_set
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.pandas import data_frame_to_smiles_tuples

logger = logging.getLogger(__name__)


def main():

    root_data_directory = "data_by_environments"

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Define the properties and environments we are interested in.
    environments_of_interest = [
        os.path.basename(x) for x in glob("data_by_environments/*")
    ]

    properties_of_interest = [
        (EnthalpyOfMixing, SubstanceType.Binary),
        (Density, SubstanceType.Binary),
        # (ExcessMolarVolume, SubstanceType.Binary),
    ]
    friendly_names = {
        (EnthalpyOfMixing, SubstanceType.Binary): "Hmix(x)",
        (Density, SubstanceType.Binary): "rho(x)",
        # (ExcessMolarVolume, SubstanceType.Binary): "Vexcess(x)",
    }

    property_combinations = [(x,) for x in properties_of_interest]
    property_combinations.extend(itertools.combinations(properties_of_interest, 2))

    data_rows = []

    for environment_of_interest in environments_of_interest:

        environment_1, environment_2 = environment_of_interest.split("_")

        data_row = {"Environment 1": environment_1, "Environment 2": environment_2}

        data_directory = os.path.join(
            root_data_directory, "_".join([environment_1, environment_2]), "all_data"
        )

        for property_combination in property_combinations:

            # Find the set of substances which are common to all of the
            # specified property types.
            all_substance_smiles = []
            property_names = []

            for property_tuple in property_combination:

                property_names.append(friendly_names[property_tuple])

                data_frame = load_processed_data_set(data_directory, *property_tuple)

                if len(data_frame) == 0:
                    all_substance_smiles = []
                    break

                substance_smiles = set(data_frame_to_smiles_tuples(data_frame))
                all_substance_smiles.append(substance_smiles)

            common_substance_smiles = {}

            if len(all_substance_smiles) > 0:
                common_substance_smiles = set.intersection(*all_substance_smiles)

            property_string = " + ".join(property_names)
            data_row[property_string] = len(common_substance_smiles)

        data_rows.append(data_row)

    columns = [
        "Environment 1",
        "Environment 2",
        *[" + ".join([friendly_names[x] for x in y]) for y in property_combinations],
    ]

    summary_frame = pandas.DataFrame(data=data_rows, columns=columns)
    summary_frame.fillna(0, inplace=True)
    summary_frame.sort_values(["Hmix(x) + rho(x)"], ascending=False, inplace=True)

    summary_frame.to_csv("summary.csv", index=False)

    with open("summary.md", "w") as file:
        summary_frame.to_markdown(file, showindex=False)


if __name__ == "__main__":
    main()
