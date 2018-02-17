# CUDA GPU code

Implementation of PFHub Benchmark 7: Method of Manufactured Solutions
with OpenMP threading and CUDA acceleration

## Usage

This directory contains a makefile with three important invocations:
 1. ```make``` will build the executable, named ```manufactured```,
    from its dependencies.
 2. ```make run``` will execute ```manufactured``` using the defaults listed in
    ```../common-manufactured/params.txt```, writing PNG and CSV output for
    inspection. ```runlog.csv``` contains the time-evolution of the weighted
    sum-of-squares residual from the analytical solution, as well as runtime
    data.
 3. ```make clean``` will remove the executable and object files ```.o```,
    but not the data.

## Dependencies

To build this code, you must have installed
 * [GNU make][_make]
 * [PNG library][_png]
 * [CUDA toolkit][_cuda]

```make``` and ```libpng``` can be installed through your operating
system's package manager, *e.g.* ```apt-get install make libpng12-dev```.
The [CUDA][_cuda] software should be installed using up-to-date distributions
from their websites since the packaged versions are often several versions
behind, and GPU hardware support evolves quickly. Note that CUDA is not
compatible with all GPU architectures. CUDA hardware can be emulated on the CPU
using the [MCUDA framework][_mcuda]. Proof-of-concept trials on GPU hardware
can be run on [Amazon's EC2][_aws] and equivalent HPC cloud computing platforms.

If you read the ```Makefile```s, you will see that this code also invokes
OpenMP (via compiler flag ```-fopenmp```). This is because some
operations &mdash; namely array allocation and application of boundary
conditions &mdash; cannot be performed efficiently on the GPU, due to the high
expense of transferring data in and out compared to the small amount of work
to be done. This program therefore implement an "OpenMP + CUDA" programming
model, where CPU threading is used to modify a few values and GPU processing
is used to perform the real work.

## Customization

The default input file ```../common-manufactured/params.txt``` defines key-value
pairs, with one pair per line. The two-character keys are predefined, and must
all be present. Descriptive comments follow the value on each line. If you wish
to change parameters, either modify ```params.txt``` in place and ```make run```,
or create your own copy of ```params.txt``` and execute
```./manufactured <your_params.txt>```. The file name and extension make
no difference, so long as it contains plain text.

### Disclaimer

Certain commercial entities, equipment, or materials may be identified in this
document in order to describe an experimental procedure or concept adequately.
Such identification is not intended to imply recommendation or endorsement by
the [National Institute of Standards and Technology (NIST)](http://www.nist.gov),
nor is it intended to imply that the entities, materials, or equipment are
necessarily the best available for the purpose.

[_aws]:    https://aws.amazon.com/ec2/Elastic-GPUs/
[_cuda]:   https://developer.nvidia.com/cuda-toolkit
[_make]:   https://www.gnu.org/software/make/
[_mcuda]:  http://impact.crhc.illinois.edu/mcuda.aspx
[_png]:    http://www.libpng.org/pub/png/libpng.html
