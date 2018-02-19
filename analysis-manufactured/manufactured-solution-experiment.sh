#!/bin/bash

# HiPerC: High Performance Computing Strategies for Boundary Value Problems
# written by Trevor Keller and available from https://github.com/usnistgov/hiperc
#
# This software was developed at the National Institute of Standards and Technology
# by employees of the Federal Government in the course of their official duties.
# Pursuant to title 17 section 105 of the United States Code this software is not
# subject to copyright protection and is in the public domain. NIST assumes no
# responsibility whatsoever for the use of this software by other parties, and makes
# no guarantees, expressed or implied, about its quality, reliability, or any other
# characteristic. We would appreciate acknowledgement if the software is used.
#
# This software can be redistributed and/or modified freely provided that any
# derivative works bear some notice that they are derived from it, and any modified
# versions bear some notice that they have been modified.
#
# Questions/comments to Trevor Keller (trevor.keller@nist.gov)

# This script will run PFHub Benchmark 7: Method of Manufactured Solutions
# through a convergence series by refining the mesh resolution, holding
# the timestep constant and allowing the CFL condition to float.
# Launch from the analysis-manufactured directory.

# Move to HiPerC root directory
DATADIR=$(pwd)
cd `dirname "${DATADIR}"`

# Back-up params.txt and clear output logfile
cp -a common-manufactured/params.txt analysis-manufactured/params.bak
rm -f gpu-cuda-manufactured/runlog.csv

for i in {1..6}
do
    kp=0.0004
    DT=0.0002
	NX=$(echo "print int(2+200*${i})"  | python2)
	NY=$(echo "print int(2+100*${i})"  | python2)
    NS=$(echo "print int(8.0/${DT})" | python2)
	DX=$(echo "print 0.01/${i}"      | python2)
    CO=$(echo "print (4.0*${kp}*${DT})/(${DX}**2)" | python2)
	sed -e "s/CFL/${CO}/" \
        -e "s/dXY/${DX}/g" \
        -e "s/KP/${kp}/" \
        -e "s/NS/${NS}/" \
        -e "s/NX/${NX}/" \
        -e "s/NY/${NY}/" \
        analysis-manufactured/params.in > common-manufactured/params.txt
	echo "Running with resolution h=${DX}:"
	make run_gpu_manufactured
	sleep 1
	echo
done


mv analysis-manufactured/params.bak common-manufactured/params.txt

cd $DATADIR

mv ../gpu-cuda-manufactured/runlog.csv spatial-convergence.csv

python3 mms-analysis.py
