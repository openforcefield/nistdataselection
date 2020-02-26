## Shared Data Curation

This directory contains the scripts used to process all data retrieved from the ThermoML archive (stored in the 
``raw_archives`` folder):
 
* ``process_data.py`` - Processes all ThermoML ``.xml`` files and stores them as ``.csv`` files in the ``processed_data`` 
  directory. Each file contains one type of property, measured for a system with a specific number of components (i.e 
  pure, binary, tenerary... etc.) **Note: The processed files have their uncertainties stripped as these have not been
  released by NIST**.
  
* ``filter_data.py`` - Filters the processed data according to the standard set of filters outlines below, and stores
  the filtered files in the ``filtered_data`` directory.
  
The processed data in this directory is expected to be referenced / loaded by all of the studies.

In addition, the ``average_uncertainties`` folder contains a script which collects information about the minimum,
maximum, mean and mode uncertainties for each processed property. **Note: This script reprocesses the ThermoML files
without stripping their uncertainties. These processed files should not be shared publicly**.

### Data Filters

The data filters only retain data which was measured for:

* temperatures in the range 288.15 - 323.15 K.
* pressures in the range 0.95 - 1.05 atm.
* substances which only contain the elements H, N, C, O, Br, Cl, F, S.
* substances whose components have their stereochemistry defined.
* substances which do not have a net charge.
* substances which do not contain ionic liquids.

Additionally the filters remove any duplicate data points, retaining the data point with the highest uncertainty if 
defined.
