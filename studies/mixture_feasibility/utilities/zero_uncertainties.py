from evaluator.datasets import PhysicalPropertyDataSet

dataset = PhysicalPropertyDataSet.from_json("selected.json")

for data in dataset:
    data.uncertainty = 0.0 * data.default_unit()

dataset.json("selected_uncertainties.json")
