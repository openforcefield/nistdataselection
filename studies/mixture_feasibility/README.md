The files contained within this folder constitute all input scripts used as part of the feasibility study [outlined
here](https://openforcefield.atlassian.net/wiki/spaces/FF/pages/122454022/Binary+Mixture+Data+Feasibility+Studies).

In particular, the:

* ``data_availability`` folder contains the scripts used to extract the sets of pure densities, binary densities,
  excess molar volumes, pure enthalpies of vaporization and enthalpies of mixing measured for substances containing
  alcohols, ester and carboxylic acids.
  
  In addition, it contains scripts to convert between excess molar volume and binary mass density.
  
* ``pure_optimization`` folder contains the scripts used to compile the training set for the study of optimizing 
  against only pure condensed phase properties, to run ``ForceBalance`` on that training set, and to benchmark against
  the test set.
  
* ``mixture_optimization`` folder contains the scripts used to compile the training set for the study of 
  optimizing against only condensed phase properties of mixture, to run ``ForceBalance`` on that training set, and to 
  benchmark against the test set.
  
* ``pure_mixture_optimization`` folder contains the scripts used to compile the training set for the study of 
  optimizing against condensed phase properties of pure and binary substances, to run ``ForceBalance`` on that 
  training set, and to benchmark against the test set.

* ``benchmark_selection`` folder contains the scripts used to compile the benchmark data set.

* ``average_uncertainties`` folder contains a script to extract the average uncertainty in each property of interest.