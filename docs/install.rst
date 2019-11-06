Installation
============

The data selection tools are currently only installable from source. It is recommended to install the
tools within a conda environment, and allow the conda package manager to install the required dependencies.

More information about conda and instructions to perform a lightweight miniconda installation `can be
found here <https://docs.conda.io/en/latest/miniconda.html>`_. It will be assumed that these have been
followed and conda is available on your machine.

Installation from Source
------------------------

To install the tools from source, clone the repository from `github
<https://github.com/openforcefield/nistdataselection>`_::

    git clone https://github.com/openforcefield/nistdataselection
    cd nistdataselection

Create a custom conda environment which contains the required dependencies and activate it::

    conda env create --name nistdataselection --file devtools/conda-envs/test_env.yaml
    conda activate nistdataselection

The final step is to install the tools themselves::

    python setup.py develop

