The purpose of the scripts within this folder is to extract all density (pure and binary), enthalpy of vaporization,
enthalpy of mixing and excess molar volume measurements made for different chemical environments.

In particular, the:

* ``convert_density_data.py`` script inter-converts all excess molar volume data into binary mass density data
  and vice versa in those cases where pure density measurements are available.
  
  **Note** To convert a given binary property we require the pure density of both components in the binary mix to
  be available at the *same state* (i.e temperature and pressure) that the binary property was measured at. Due to
  precision and rounding issues, we round all pressures to 1 decimal place (in kPa) and all temperatures to 2 decimal
  places (in K) when comparing if two states are the same.
  
  The converted data is stored in the ``converted_density_data`` folder.
  
* ``filter_data.py`` applies filters to the the processed data in the ``shared`` and `converted_density_data`
  folders, retaining only those measurements made for components containing a specified set of chemical moieties, 
  and for substances only composed of C, H and O elements. 
  
  The output is partitioned into the  `data_by_environments/{environment_1}_{environment_2}/all_data` folders.

* ``find_common_data.py`` script finds those substances for which their is data available for multiple types of property 
  (i.e those substances which have both binary mass density and enthalpy of mixing data available).

  All data for such substances is extracted and partitioned into the `data_by_environments/{environment_1}_{environment_2}/common_data` 
  folders.
  
  **Note** This script does *not* check whether the different types of properties were measured at the same states (i.e
  temperature, pressure and mole fraction).
  
* ``source_h_vap_data.py`` scripts builds a new data set of enthalpy of vaporization measurements source from the
  literature. The output set is stored in the ``sourced_h_vap_data`` folder.
