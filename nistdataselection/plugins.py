"""
Extends the number of properties which can be understood and parsed by
the `evaluator.datasets.ThermoMLDataSet` object.
"""

from evaluator.datasets import PhysicalProperty, PropertyPhase
from evaluator.datasets.thermoml import thermoml_property


@thermoml_property("Specific volume, m3/kg", supported_phases=PropertyPhase.Liquid)
class SpecificVolume(PhysicalProperty):
    pass


@thermoml_property("Amount density, mol/m3", supported_phases=PropertyPhase.Liquid)
class AmountDensity(PhysicalProperty):
    pass


@thermoml_property("Molar volume, m3/mol", supported_phases=PropertyPhase.Liquid)
class MolarVolume(PhysicalProperty):
    pass


@thermoml_property(
    "Partial molar volume, m3/mol", supported_phases=PropertyPhase.Liquid
)
class PartialMolarVolume(PhysicalProperty):
    pass


@thermoml_property(
    "Apparent molar enthalpy, kJ/mol", supported_phases=PropertyPhase.Liquid
)
class ApparentMolarEnthalpy(PhysicalProperty):
    pass


@thermoml_property(
    "Molar enthalpy of solution, kJ/mol", supported_phases=PropertyPhase.Liquid
)
class MolarEnthalpyOfSolution(PhysicalProperty):
    pass


@thermoml_property(
    "Molar enthalpy of dilution, kJ/mol", supported_phases=PropertyPhase.Liquid
)
class MolarEnthalpyOfDilution(PhysicalProperty):
    pass


@thermoml_property(
    "Molar enthalpy of mixing with solvent, kJ/mol",
    supported_phases=PropertyPhase.Liquid.Gas,
)
class EnthalpyOfMixingWithSolvent(PhysicalProperty):
    pass


@thermoml_property("Activity coefficient", supported_phases=PropertyPhase.Liquid)
class ActivityCoefficient(PhysicalProperty):
    pass


@thermoml_property(
    "Mean ionic activity, (mol/dm3)^n", supported_phases=PropertyPhase.Liquid
)
class MeanIonicActivity(PhysicalProperty):
    pass


@thermoml_property(
    "Mean ionic activity coefficient", supported_phases=PropertyPhase.Liquid
)
class MeanIonicActivityCoefficient(PhysicalProperty):
    pass


@thermoml_property("Osmotic pressure, kPa", supported_phases=PropertyPhase.Liquid)
class OsmoticPressure(PhysicalProperty):
    pass


@thermoml_property("Osmotic coefficient", supported_phases=PropertyPhase.Liquid)
class OsmoticCoefficient(PhysicalProperty):
    pass


@thermoml_property(
    "Molar heat capacity at constant pressure, J/K/mol",
    supported_phases=PropertyPhase.Liquid,
)
class IsobaricMolarHeatCapacity(PhysicalProperty):
    pass


@thermoml_property(
    "Specific heat capacity at constant pressure, J/K/kg",
    supported_phases=PropertyPhase.Liquid,
)
class IsobaricSpecificHeatCapacity(PhysicalProperty):
    pass


@thermoml_property(
    "Heat capacity at constant pressure per volume, J/K/m3",
    supported_phases=PropertyPhase.Liquid,
)
class IsobaricHeatCapacity(PhysicalProperty):
    pass


@thermoml_property(
    "Molar heat capacity at constant volume, J/K/mol",
    supported_phases=PropertyPhase.Liquid,
)
class IsochoricMolarHeatCapacity(PhysicalProperty):
    pass


@thermoml_property(
    "Specific heat capacity at constant volume, J/K/kg",
    supported_phases=PropertyPhase.Liquid,
)
class IsochoricSpecificHeatCapacity(PhysicalProperty):
    pass


@thermoml_property(
    "Heat capacity at constant volume per volume, J/K/m3",
    supported_phases=PropertyPhase.Liquid,
)
class IsochoricHeatCapacity(PhysicalProperty):
    pass


@thermoml_property(
    "Apparent molar heat capacity, J/K/mol", supported_phases=PropertyPhase.Liquid
)
class ApparentMolarHeatCapacity(PhysicalProperty):
    pass


@thermoml_property(
    "Isothermal compressibility, 1/kPa", supported_phases=PropertyPhase.Liquid
)
class IsothermalCompressibility(PhysicalProperty):
    pass


@thermoml_property(
    "Excess isothermal compressibility, 1/kPa", supported_phases=PropertyPhase.Liquid
)
class ExcessIsothermalCompressibility(PhysicalProperty):
    pass


@thermoml_property("Compressibility factor", supported_phases=PropertyPhase.Liquid)
class CompressibilityFactor(PhysicalProperty):
    pass


@thermoml_property(
    "Isobaric coefficient of expansion, 1/K", supported_phases=PropertyPhase.Liquid
)
class IsobaricCoefficientOfExpansion(PhysicalProperty):
    pass


@thermoml_property("Speed of sound, m/s", supported_phases=PropertyPhase.Liquid)
class SpeedOfSound(PhysicalProperty):
    pass


@thermoml_property("Excess speed of sound, m/s", supported_phases=PropertyPhase.Liquid)
class ExcessSpeedOfSound(PhysicalProperty):
    pass


@thermoml_property("2nd virial coefficient, m3/mol", supported_phases=PropertyPhase.Gas)
class SecondVirialCoefficient(PhysicalProperty):
    pass


@thermoml_property(
    "Surface tension liquid-gas, N/m",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class LiquidGasSurfaceTension(PhysicalProperty):
    pass


@thermoml_property(
    "Henry's Law constant (mole fraction scale), kPa",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class HenrysLawConstantMoleFractionScale(PhysicalProperty):
    pass


@thermoml_property(
    "Henry's Law constant (molality scale), kPa*kg/mol",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class HenrysLawConstantMolalityScale(PhysicalProperty):
    pass


@thermoml_property(
    "Henry's Law constant (amount concentration scale), kPa*dm3/mol",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class HenrysLawConstantAmountConcentrationScale(PhysicalProperty):
    pass


@thermoml_property("(Relative) activity", supported_phases=PropertyPhase.Liquid)
class RelativeActivity(PhysicalProperty):
    pass


@thermoml_property(
    "Vapor or sublimation pressure, kPa",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class VaporPressure(PhysicalProperty):
    pass
