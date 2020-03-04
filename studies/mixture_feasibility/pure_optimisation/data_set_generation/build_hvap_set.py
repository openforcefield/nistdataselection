from evaluator import unit
from evaluator.datasets import MeasurementSource, PhysicalPropertyDataSet, PropertyPhase
from evaluator.properties import EnthalpyOfVaporization
from evaluator.substances import Substance
from evaluator.thermodynamics import ThermodynamicState


def main():

    # Build a data set containing the training set Hvap measurements sourced
    # from the literature.
    h_vap_data_set = PhysicalPropertyDataSet()
    h_vap_phase = PropertyPhase(PropertyPhase.Liquid | PropertyPhase.Gas)

    h_vap_data_set.add_properties(
        # Formic Acid
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("OC=O"),
            value=46.3 * unit.kilojoule / unit.mole,
            uncertainty=0.25 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.3891/acta.chem.scand.24-2612"),
        ),
        # Acetic Acid
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(O)=O"),
            value=51.6 * unit.kilojoule / unit.mole,
            uncertainty=0.75 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.3891/acta.chem.scand.24-2612"),
        ),
        # Propionic Acid
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCC(O)=O"),
            value=55 * unit.kilojoule / unit.mole,
            uncertainty=1 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.3891/acta.chem.scand.24-2612"),
        ),
        # Butyric Acid
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCC(O)=O"),
            value=58 * unit.kilojoule / unit.mole,
            uncertainty=2 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.3891/acta.chem.scand.24-2612"),
        ),
        # Isobutyric Acid
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(C)C(O)=O"),
            value=53 * unit.kilojoule / unit.mole,
            uncertainty=2 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.3891/acta.chem.scand.24-2612"),
        ),
        # Water
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("O"),
            value=43.98 * unit.kilojoule / unit.mole,
            uncertainty=0.02199 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/S0021-9614(71)80108-8"),
        ),
        # Methanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CO"),
            value=37.83 * unit.kilojoule / unit.mole,
            uncertainty=0.11349 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0378-3812(85)90026-3"),
        ),
        # Ethanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCO"),
            value=42.46 * unit.kilojoule / unit.mole,
            uncertainty=0.12738 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0378-3812(85)90026-3"),
        ),
        # 1-Propanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCO"),
            value=47.5 * unit.kilojoule / unit.mole,
            uncertainty=0.1425 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0378-3812(85)90026-3"),
        ),
        # Isopropanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(C)O"),
            value=45.48 * unit.kilojoule / unit.mole,
            uncertainty=0.13644 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0378-3812(85)90026-3"),
        ),
        # n-Butanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCO"),
            value=52.42 * unit.kilojoule / unit.mole,
            uncertainty=0.15726 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0378-3812(85)90026-3"),
        ),
        # Isobutanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(C)CO"),
            value=50.89 * unit.kilojoule / unit.mole,
            uncertainty=0.15267 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0378-3812(85)90026-3"),
        ),
        # 2-Butanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCC(C)O"),
            value=49.81 * unit.kilojoule / unit.mole,
            uncertainty=0.14943 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0378-3812(85)90026-3"),
        ),
        # t-butanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(C)(C)O"),
            value=46.75 * unit.kilojoule / unit.mole,
            uncertainty=0.14025 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0378-3812(85)90026-3"),
        ),
        # n-pentanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCCO"),
            value=44.36 * unit.kilojoule / unit.mole,
            uncertainty=0.13308 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0378-3812(85)90026-3"),
        ),
        # 1-hexanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCCCO"),
            value=61.85 * unit.kilojoule / unit.mole,
            uncertainty=0.2 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(77)90202-6"),
        ),
        # 1-heptanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCCCCO"),
            value=66.81 * unit.kilojoule / unit.mole,
            uncertainty=0.2 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(77)90202-6"),
        ),
        # 1-octanol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCCCCCO"),
            value=70.98 * unit.kilojoule / unit.mole,
            uncertainty=0.42 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(77)90202-6"),
        ),
        # Propyl formate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCOC=O"),
            value=37.49 * unit.kilojoule / unit.mole,
            uncertainty=0.07498 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19803233"),
        ),
        # Butyl formate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCOC=O"),
            value=41.25 * unit.kilojoule / unit.mole,
            uncertainty=0.0825 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19803233"),
        ),
        # Methyl acetate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("COC(C)=O"),
            value=32.3 * unit.kilojoule / unit.mole,
            uncertainty=0.0646 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19803233"),
        ),
        # Ethyl acetate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCOC(C)=O"),
            value=35.62 * unit.kilojoule / unit.mole,
            uncertainty=0.07124 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19803233"),
        ),
        # Propyl acetate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCOC(C)=O"),
            value=39.83 * unit.kilojoule / unit.mole,
            uncertainty=0.07966 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19803233"),
        ),
        # Methyl propionate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCC(=O)OC"),
            value=35.85 * unit.kilojoule / unit.mole,
            uncertainty=0.0717 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19803233"),
        ),
        # Ethyl propionate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCOC(=O)CC"),
            value=39.25 * unit.kilojoule / unit.mole,
            uncertainty=0.0785 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19803233"),
        ),
        # Butyl acetate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=313.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCOC(C)=O"),
            value=42.96 * unit.kilojoule / unit.mole,
            uncertainty=0.08592 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19803233"),
        ),
        # Propyl propionate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=313.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCOC(=O)CC"),
            value=42.14 * unit.kilojoule / unit.mole,
            uncertainty=0.08428 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19803233"),
        ),
        # Methyl Butanoate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCC(=O)OC"),
            value=40.1 * unit.kilojoule / unit.mole,
            uncertainty=0.4 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1007/BF00653098"),
        ),
        # Methyl Pentanoate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCC(=O)OC"),
            value=44.32 * unit.kilojoule / unit.mole,
            uncertainty=0.5 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1007/BF00653098"),
        ),
        # Ethyl Butanoate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCC(=O)OCC"),
            value=42.86 * unit.kilojoule / unit.mole,
            uncertainty=0.1 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(86)90070-4"),
        ),
        # Ethylene glycol diacetate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(=O)OCCOC(=O)C"),
            value=61.44 * unit.kilojoule / unit.mole,
            uncertainty=0.15 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(86)90070-4"),
        ),
        # Methyl formate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=293.25 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("COC=O"),
            value=28.7187400224 * unit.kilojoule / unit.mole,
            uncertainty=0.0 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19760001"),
        ),
        # Ethyl formate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=304 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCOC=O"),
            value=31.63314346416 * unit.kilojoule / unit.mole,
            uncertainty=0.0 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1135/cccc19760001"),
        ),
        # 1,3-propanediol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("OCCCO"),
            value=70.5 * unit.kilojoule / unit.mole,
            uncertainty=0.3 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1021/je060419q"),
        ),
        # 2,4 pentanediol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(CC(C)O)O"),
            value=72.5 * unit.kilojoule / unit.mole,
            uncertainty=0.3 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1021/je060419q"),
        ),
        # 2-Me-2,4-pentanediol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(O)CC(C)(C)O"),
            value=68.9 * unit.kilojoule / unit.mole,
            uncertainty=0.4 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1021/je060419q"),
        ),
        # 2,2,4-triMe-1,3-pentanediol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(C)C(O)C(C)(C)CO"),
            value=75.3 * unit.kilojoule / unit.mole,
            uncertainty=0.5 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1021/je060419q"),
        ),
        # glycerol
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("C(C(CO)O)O"),
            value=91.7 * unit.kilojoule / unit.mole,
            uncertainty=0.9 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(88)90173-5"),
        ),
        # Diethyl Malonate
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCOC(=O)CC(=O)OCC"),
            value=61.70 * unit.kilojoule / unit.mole,
            uncertainty=0.25 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1021/je100231g"),
        ),
    )

    h_vap_data_set.to_pandas().to_csv("alcohol_ester_h_vap.csv", index=False)


if __name__ == "__main__":
    main()
