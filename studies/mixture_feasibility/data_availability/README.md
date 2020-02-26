The purpose of the scripts within this directory is to extract all density (pure and binary), enthalpy of vaporization,
enthalpy of mixing and excess molar volume measurements made for alcohols, esters and acids.

In particular, the:
  
* ``filter_alcohol_ester.py`` applies filters to the the processed data in the ``shared`` folder, retaining only those
  measurements made for alcohols, esters and acids, and for substances only composed of C, H and O elements. The output 
  is stored in the ``all_alcohol_ester_data`` folder.

* ``convert_density_data.py`` script inter-converts all excess molar volume data into binary mass density data
  and vice versa in those cases where pure density measurements are available.
  
  **Note** To convert a given binary property we require the pure density of both components in the binary mix to
  be available at the *same state* (i.e temperature and pressure) that the binary property was measured at. Due to
  precision and rounding issues, we round all pressures to 1 decimal place (in kPa) and all temperatures to 2 decimal
  places (in K) when comparing if two states are the same.
  
  The converted data is stored in the ``converted_density_data`` folder.