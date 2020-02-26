"""
Extends the number of properties which can be understood and parsed by
the `evaluator.datasets.ThermoMLDataSet` object.
"""
from evaluator import unit
from evaluator.datasets import PhysicalProperty, PropertyPhase
from evaluator.datasets.thermoml import thermoml_property


@thermoml_property("Specific volume, m3/kg", supported_phases=PropertyPhase.Liquid)
class SpecificVolume(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.meter ** 3 / unit.kilogram


@thermoml_property("Amount density, mol/m3", supported_phases=PropertyPhase.Liquid)
class AmountDensity(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.mol / unit.meter ** 3


@thermoml_property("Molar volume, m3/mol", supported_phases=PropertyPhase.Liquid)
class MolarVolume(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.meter ** 3 / unit.mole


@thermoml_property(
    "Partial molar volume, m3/mol", supported_phases=PropertyPhase.Liquid
)
class PartialMolarVolume(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.meter ** 3 / unit.mole


@thermoml_property(
    "Apparent molar enthalpy, kJ/mol", supported_phases=PropertyPhase.Liquid
)
class ApparentMolarEnthalpy(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.kilojoule / unit.mole


@thermoml_property(
    "Molar enthalpy of solution, kJ/mol", supported_phases=PropertyPhase.Liquid
)
class MolarEnthalpyOfSolution(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.kilojoule / unit.mole


@thermoml_property(
    "Molar enthalpy of dilution, kJ/mol", supported_phases=PropertyPhase.Liquid
)
class MolarEnthalpyOfDilution(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.kilojoule / unit.mole


@thermoml_property(
    "Molar enthalpy of mixing with solvent, kJ/mol",
    supported_phases=PropertyPhase.Liquid.Gas,
)
class EnthalpyOfMixingWithSolvent(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.kilojoule / unit.mole


@thermoml_property("Activity coefficient", supported_phases=PropertyPhase.Liquid)
class ActivityCoefficient(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.dimensionless


@thermoml_property(
    "Mean ionic activity coefficient", supported_phases=PropertyPhase.Liquid
)
class MeanIonicActivityCoefficient(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.dimensionless


@thermoml_property("Osmotic pressure, kPa", supported_phases=PropertyPhase.Liquid)
class OsmoticPressure(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.kilopascal


@thermoml_property("Osmotic coefficient", supported_phases=PropertyPhase.Liquid)
class OsmoticCoefficient(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.dimensionless


@thermoml_property(
    "Molar heat capacity at constant pressure, J/K/mol",
    supported_phases=PropertyPhase.Liquid,
)
class IsobaricMolarHeatCapacity(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.joule / unit.kelvin / unit.mole


@thermoml_property(
    "Specific heat capacity at constant pressure, J/K/kg",
    supported_phases=PropertyPhase.Liquid,
)
class IsobaricSpecificHeatCapacity(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.joule / unit.kelvin / unit.kilogram


@thermoml_property(
    "Heat capacity at constant pressure per volume, J/K/m3",
    supported_phases=PropertyPhase.Liquid,
)
class IsobaricHeatCapacity(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.joule / unit.kelvin / unit.meter ** 3


@thermoml_property(
    "Molar heat capacity at constant volume, J/K/mol",
    supported_phases=PropertyPhase.Liquid,
)
class IsochoricMolarHeatCapacity(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.joule / unit.kelvin / unit.mole


@thermoml_property(
    "Specific heat capacity at constant volume, J/K/kg",
    supported_phases=PropertyPhase.Liquid,
)
class IsochoricSpecificHeatCapacity(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.joule / unit.kelvin / unit.kilogram


@thermoml_property(
    "Heat capacity at constant volume per volume, J/K/m3",
    supported_phases=PropertyPhase.Liquid,
)
class IsochoricHeatCapacity(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.joule / unit.kelvin / unit.meter ** 3


@thermoml_property(
    "Apparent molar heat capacity, J/K/mol", supported_phases=PropertyPhase.Liquid
)
class ApparentMolarHeatCapacity(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.joule / unit.kelvin / unit.mole


@thermoml_property(
    "Isothermal compressibility, 1/kPa", supported_phases=PropertyPhase.Liquid
)
class IsothermalCompressibility(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return (1.0 / unit.kilopascal).units


@thermoml_property(
    "Excess isothermal compressibility, 1/kPa", supported_phases=PropertyPhase.Liquid
)
class ExcessIsothermalCompressibility(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return (1.0 / unit.kilopascal).units


@thermoml_property("Compressibility factor", supported_phases=PropertyPhase.Liquid)
class CompressibilityFactor(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.dimensionless


@thermoml_property(
    "Isobaric coefficient of expansion, 1/K", supported_phases=PropertyPhase.Liquid
)
class IsobaricCoefficientOfExpansion(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return (1.0 / unit.kelvin).units


@thermoml_property("Speed of sound, m/s", supported_phases=PropertyPhase.Liquid)
class SpeedOfSound(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.meter / unit.second


@thermoml_property("Excess speed of sound, m/s", supported_phases=PropertyPhase.Liquid)
class ExcessSpeedOfSound(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.meter / unit.second


@thermoml_property("2nd virial coefficient, m3/mol", supported_phases=PropertyPhase.Gas)
class SecondVirialCoefficient(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.meter ** 3 / unit.mole


@thermoml_property(
    "Surface tension liquid-gas, N/m",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class LiquidGasSurfaceTension(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.newton / unit.meter


@thermoml_property(
    "Henry's Law constant (mole fraction scale), kPa",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class HenrysLawConstantMoleFractionScale(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.kilopascal


@thermoml_property(
    "Henry's Law constant (molality scale), kPa*kg/mol",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class HenrysLawConstantMolalityScale(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.kilopascal * unit.kilogram / unit.mole


@thermoml_property(
    "Henry's Law constant (amount concentration scale), kPa*dm3/mol",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class HenrysLawConstantAmountConcentrationScale(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.kilopascal * unit.decimeter ** 3 / unit.mole


@thermoml_property("(Relative) activity", supported_phases=PropertyPhase.Liquid)
class RelativeActivity(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.dimensionless


@thermoml_property(
    "Vapor or sublimation pressure, kPa",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class VaporPressure(PhysicalProperty):
    @classmethod
    def default_unit(cls):
        return unit.kilopascal
