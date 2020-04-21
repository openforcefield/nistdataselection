"""The scripts will produce a pandas `DataFrame` which contains information
about which combinations of environments have which data types.
"""
import itertools
import logging
import os
from collections import defaultdict

import pandas
from evaluator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)

from nistdataselection.curation.filtering import filter_by_checkmol
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import (
    chemical_environment_codes,
    substance_type_to_int,
)

logger = logging.getLogger(__name__)


def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    data_set_path = os.path.join("test_sets", "full_set.csv")
    data_set = pandas.read_csv(data_set_path)

    # Define the environments we are interested in.
    environments = {
        "alcohol": [
            chemical_environment_codes["hydroxy"],
            chemical_environment_codes["alcohol"],
        ],
        "ester": [
            chemical_environment_codes["caboxylic_acid"],
            chemical_environment_codes["ester"],
        ],
    }

    # Define the properties and environments we are interested in.
    properties_of_interest = [
        (Density, SubstanceType.Pure),
        (EnthalpyOfVaporization, SubstanceType.Pure),
        (EnthalpyOfMixing, SubstanceType.Binary),
        (Density, SubstanceType.Binary),
        (ExcessMolarVolume, SubstanceType.Binary),
    ]

    friendly_names = {
        (Density, SubstanceType.Pure): "rho",
        (EnthalpyOfVaporization, SubstanceType.Pure): "Hvap",
        (EnthalpyOfMixing, SubstanceType.Binary): "Hmix(x)",
        (Density, SubstanceType.Binary): "rho(x)",
        (ExcessMolarVolume, SubstanceType.Binary): "Vexcess(x)",
    }

    # Extend the combination of properties.
    property_combinations = [(x,) for x in properties_of_interest]

    binary_properties = [
        x for x in properties_of_interest if x[1] == SubstanceType.Binary
    ]
    property_combinations.extend(itertools.combinations(binary_properties, 2))

    # Define the combinations of environments
    environment_combinations = {
        SubstanceType.Pure: [(x,) for x in environments],
        SubstanceType.Binary: [
            *[(x, x) for x in environments],
            *itertools.combinations(environments, 2),
        ],
    }

    data_counts = defaultdict(dict)

    for property_types in property_combinations:

        all_substances = defaultdict(list)

        for property_type, substance_type in property_types:

            header = (
                f"{property_type.__name__} Value ({property_type.default_unit():~})"
            )

            n_components = substance_type_to_int[substance_type]

            property_data = data_set[data_set[header].notnull()]
            property_data = property_data[property_data["N Components"] == n_components]

            for environment_types in environment_combinations[substance_type]:

                environment_types = tuple(sorted(environment_types))
                chemical_environments = [environments[x] for x in environment_types]

                environment_data = filter_by_checkmol(
                    property_data, *chemical_environments
                )

                components = []

                for index in range(n_components):
                    components.append([*environment_data[f"Component {index + 1}"]])

                all_substances[environment_types].append(
                    set(tuple(sorted(x)) for x in zip(*components))
                )

        common_substances = {x: set.intersection(*y) for x, y in all_substances.items()}

        for environment_type in common_substances:

            data_counts[environment_type][property_types] = len(
                common_substances[environment_type]
            )

    data_rows = []

    for environment_types in data_counts:

        data_row = {}

        for index, environment_type in enumerate(environment_types):
            data_row[f"Environment {index + 1}"] = environment_type

        for property_types in data_counts[environment_types]:

            header = " + ".join(friendly_names[x] for x in property_types)
            data_row[header] = int(data_counts[environment_types][property_types])

        data_rows.append(data_row)

    n_environments = max(len(x) for x in data_counts)

    count_frame = pandas.DataFrame(data_rows)
    reordered_frame = pandas.DataFrame()

    # Re-order the headers.
    for index in range(n_environments):

        reordered_frame[f"Environment {index + 1}"] = count_frame[
            f"Environment {index + 1}"
        ]

    for column_name in count_frame:

        if "Environment" in column_name:
            continue

        reordered_frame[column_name] = count_frame[column_name]

    reordered_frame.fillna("-", inplace=True)
    reordered_frame.to_csv("summary.csv", index=False)

    with open("summary.md", "w") as file:
        reordered_frame.to_markdown(file, showindex=False)


if __name__ == "__main__":
    main()
