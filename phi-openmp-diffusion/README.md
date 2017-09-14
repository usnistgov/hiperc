# Xeon Phi code

This directory contains implementations of the diffusion equation for KNL, with help from the following text:
> Jeffers, Reinders, and Sodani. *Intel Xeon Phi Processor High Performance Programming: Knights Landing Edition,* 2 Ed. Morgan Kaufmann. New York: 2016.

## Usage

This directory contains a makefile with three important invocations:
 1. ```make``` will build the executable, named ```diffusion```, from its
    dependencies.
 2. ```make run``` will execute ```diffusion``` using the defaults listed in
    ```../common-diffusion/params.txt```, writing PNG and CSV output for
    inspection. ```runlog.csv``` contains the time-evolution of the weighted
    sum-of-squares residual from the analytical solution, as well as runtime
    data.
 3. ```make clean``` will remove the executable and object files ```.o```,
    but not the data.

## Dependencies

To build this code, you must have installed
 * [GNU make][_make]
 * [PNG library][_png]
 * [Intel compiler][_intel]

These are usually available through the package manager. For example,
```apt-get install make libpng12-dev```. The Intel compiler is proprietary,
but available to open-source developers at no cost. Hardware capable of AVX-512
instructions, whether a primary Xeon CPU or a [Xeon Phi][_knl], is also required.
Proof-of-concept KNL code can be run on [Rescale's ScaleX][_scalex] and
equivalent HPC cloud computing platforms. [KNL][_knl] nodes are available on
advanced research computing platforms, including Argonne National Labs'
[Bebop][_bebop], NERSC [Cori][_cori], TACC [Stampede2][_tacc], and
[XSEDE][_xsede].

## Customization

The default input file ```../common-diffusion/params.txt``` defines key-value
pairs, with one pair per line. The two-character keys are predefined, and must
all be present. Descriptive comments follow the value on each line. If you wish
to change parameters (D, runtime, etc.), either modify ```params.txt``` in
place and ```make run```, or create your own copy of ```params.txt``` and
execute ```./diffusion <your_params.txt>```. The file name and extension make
no difference, so long as it contains plain text.

### Disclaimer

Certain commercial entities, equipment, or materials may be identified in this
document in order to describe an experimental procedure or concept adequately.
Such identification is not intended to imply recommendation or endorsement by
the [National Institute of Standards and Technology (NIST)](http://www.nist.gov),
nor is it intended to imply that the entities, materials, or equipment are
necessarily the best available for the purpose.

[_bebop]:  http://www.lcrc.anl.gov/systems/resources/bebop/
[_cori]:   http://www.nersc.gov/users/computational-systems/cori/
[_gcc]:    https://gcc.gnu.org
[_intel]:  https://software.intel.com/en-us/intel-compilers
[_knl]:    https://www.intel.com/content/www/us/en/products/processors/xeon-phi/xeon-phi-processors.html
[_make]:   https://www.gnu.org/software/make/
[_png]:    http://www.libpng.org/pub/png/libpng.html
[_scalex]: http://www.rescale.com/products/
[_tacc]:   https://www.tacc.utexas.edu/systems/stampede2
[_xsede]:  https://www.xsede.org/ecosystem/resources
