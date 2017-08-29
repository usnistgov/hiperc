# Documentation

This directory contains documentation for https://github.com/usnistgov/phasefield-accelerator-benchmarks.

## Dependencies

To build this code, the following software packages must be installed.

### [Make][_make]
Usually available through the package manager. For example,
```apt-get install make``` or ```yum install make```.

### [Doxygen][_doxygen]
Usually available through the package manager. For example,
```apt-get install doxygen``` or ```yum install doxygen```.

### [Sphinx][_sphinx]
A Python package, usually available through your Python distribution's package
manager. For example, ```apt-get install python-sphinx```,
```conda install sphinx```, or ```pip install sphinx```.

### [Breathe][_breathe]
A Python package, available through [PyPI][_pypi]. For example,
```pip install breathe```.

## Usage

This directory contains a makefile with two important invocations:
 1. ```make``` will build the PDF documentation, named
    ```phasefield-accelerator-benchmarks_guide.pdf```, from the source code.
 2. ```make sphinx``` will convert the XML output from Doxygen into HTML
    compatible with [readthedocs][_rtd].
 3. ```make clean``` will remove the [Doxygen][_doxygen] and [Sphinx][_sphinx]
    build directories.

[_breathe]: https://breathe.readthedocs.io
[_doxygen]: http://www.stack.nl/~dimitri/doxygen
[_make]:    https://www.gnu.org/software/make
[_pypi]:    https://pypi.python.org/pypi
[_rtd]:     https://readthedocs.org
[_sphinx]:  http://www.sphinx-doc.org
