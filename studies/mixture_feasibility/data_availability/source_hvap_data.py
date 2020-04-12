import logging
import os

from evaluator import unit
from evaluator.attributes import UNDEFINED
from evaluator.datasets import MeasurementSource, PhysicalPropertyDataSet, PropertyPhase
from evaluator.properties import EnthalpyOfVaporization
from evaluator.substances import Substance
from evaluator.thermodynamics import ThermodynamicState

from nistdataselection.curation.filtering import filter_undefined_stereochemistry
from nistdataselection.processing import save_processed_data_set
from nistdataselection.utils import SubstanceType
from nistdataselection.utils.utils import data_frame_to_pdf

logger = logging.getLogger(__name__)


def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO)

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
            uncertainty=UNDEFINED,
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
            uncertainty=UNDEFINED,
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
        # 1,4-dioxane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("C1COCCO1"),
            value=38.64 * unit.kilojoule / unit.mole,
            uncertainty=0.05 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1039/P29820000565"),
        ),
        # oxane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("C1CCOCC1"),
            value=34.94 * unit.kilojoule / unit.mole,
            uncertainty=0.84 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1039/TF9615702125"),
        ),
        # methyl tert butyl ether
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("COC(C)(C)C"),
            value=32.42 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1016/0021-9614(80)90152-4"),
        ),
        # diisopropyl ether
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(C)OC(C)C"),
            value=32.12 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1016/0021-9614(80)90152-4"),
        ),
        # Dibutyl ether
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCOCCCC"),
            value=44.99 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1016/0021-9614(80)90152-4"),
        ),
        # cyclopentanone
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.16 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("O=C1CCCC1"),
            value=42.63 * unit.kilojoule / unit.mole,
            uncertainty=0.42 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1002/hlca.19720550510"),
        ),
        # 2-pentanone
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCC(C)=O"),
            value=38.43 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1016/0021-9614(83)90091-5"),
        ),
        # cyclohexanone
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.16 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("O=C1CCCCC1"),
            value=44.89 * unit.kilojoule / unit.mole,
            uncertainty=0.63 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1002/hlca.19720550510"),
        ),
        # cycloheptanone
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.16 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("O=C1CCCCCC1"),
            value=49.54 * unit.kilojoule / unit.mole,
            uncertainty=0.63 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1002/hlca.19720550510"),
        ),
        # cyclohexane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("C1CCCCC1"),
            value=33.02 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1135/cccc19790637"),
        ),
        # hexane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCCC"),
            value=31.55 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1135/cccc19790637"),
        ),
        # methylcyclohexane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC1CCCCC1"),
            value=35.38 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1135/cccc19790637"),
        ),
        # heptane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCCCC"),
            value=36.58 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1135/cccc19790637"),
        ),
        # iso-octane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(C)CC(C)(C)C"),
            value=35.13 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1135/cccc19790637"),
        ),
        # decane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCCCCCCC"),
            value=51.35 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.3891/acta.chem.scand.20-0536"),
        ),
        # acetone
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=300.4 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(C)=O"),
            value=30.848632 * unit.kilojoule / unit.mole,
            uncertainty=0.008368 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1021/ja01559a015"),
        ),
        # butan-2-one
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCC(C)=O"),
            value=34.51 * unit.kilojoule / unit.mole,
            uncertainty=0.04 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="0021-9614(79)90127-7"),
        ),
        # pentan-3-one
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCC(=O)CC"),
            value=38.52 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1016/0021-9614(83)90091-5"),
        ),
        # 4-methylpentan-2-one
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CC(=O)CC(C)C"),
            value=40.56 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1016/0021-9614(83)90091-5"),
        ),
        # 3-hexanone
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCC(=O)CC"),
            value=42.45 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1016/0021-9614(83)90091-5"),
        ),
        # 2-methylheptane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCCC(C)C"),
            value=39.66 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1135/cccc19790637"),
        ),
        # 3-methylpentane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCC(C)CC"),
            value=30.26 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1135/cccc19790637"),
        ),
        # 2-Methylhexane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCC(C)C"),
            value=34.85 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1135/cccc19790637"),
        ),
        # 2,3-Dimethylpentane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCC(C)C(C)C"),
            value=34.25 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1135/cccc19790637"),
        ),
        # Octane
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCCCCC"),
            value=41.47 * unit.kilojoule / unit.mole,
            uncertainty=UNDEFINED,
            source=MeasurementSource(doi="10.1135/cccc19790637"),
        ),
        # Methyl Propyl Ether
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCOC"),
            value=27.57 * unit.kilojoule / unit.mole,
            uncertainty=0.068925 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(80)90152-4"),
        ),
        # Ethyl isopropyl ether
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCOC(C)C"),
            value=30.04 * unit.kilojoule / unit.mole,
            uncertainty=0.0751 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(80)90152-4"),
        ),
        # Dipropyl ether
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCOCCC"),
            value=35.68 * unit.kilojoule / unit.mole,
            uncertainty=0.0892 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(80)90152-4"),
        ),
        # butyl methyl ether
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("CCCCOC"),
            value=32.43 * unit.kilojoule / unit.mole,
            uncertainty=0.081075 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(80)90152-4"),
        ),
        # methyl isopropyl ether
        EnthalpyOfVaporization(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
            ),
            phase=h_vap_phase,
            substance=Substance.from_components("COC(C)C"),
            value=26.41 * unit.kilojoule / unit.mole,
            uncertainty=0.066025 * unit.kilojoule / unit.mole,
            source=MeasurementSource(doi="10.1016/0021-9614(80)90152-4"),
        ),
    )

    output_directory = "sourced_h_vap_data"
    os.makedirs(output_directory, exist_ok=True)

    data_frame = h_vap_data_set.to_pandas()

    # Check for undefined stereochemistry
    filtered_data_frame = filter_undefined_stereochemistry(data_frame)

    filtered_components = {*data_frame["Component 1"]} - {
        *filtered_data_frame["Component 1"]
    }
    logger.info(
        f"Compounds without stereochemistry were removed: {filtered_components}"
    )

    save_processed_data_set(
        output_directory,
        filtered_data_frame,
        EnthalpyOfVaporization,
        SubstanceType.Pure,
    )

    file_path = os.path.join(output_directory, "enthalpy_of_vaporization_pure.pdf")
    data_frame_to_pdf(filtered_data_frame, file_path)


if __name__ == "__main__":
    main()
