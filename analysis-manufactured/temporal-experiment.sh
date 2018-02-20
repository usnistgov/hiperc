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

for i in {0..9}
do
    BX=8
    kp=0.0004
	NX=202
    NS=$(echo "print 10000+10000*${i}" | python2)
    DT=$(echo "print 8.0/${NS}" | python2)
	DX=$(echo "print 1.0/(${NX}-2)"  | python2)
	NU=$(echo "print 4.*${kp}*${DT}" | python2)
    CO=$(echo "print ${NU}/${DX}**2" | python2)
	NY=$(echo "print 2+(${NX}-2)/2"  | python2)
	sed -e "s/BXY/${BX}/g" \
        -e "s/CFL/${CO}/"  \
        -e "s/dXY/${DX}/g" \
        -e "s/KP/${KP}/"   \
        -e "s/NS/${NS}/"   \
        -e "s/NX/${NX}/"   \
        -e "s/NY/${NY}/"   \
        analysis-manufactured/params.in > common-manufactured/params.txt
	echo "Running with resolution k=${DT}:"
	make run_gpu_manufactured
	sleep 1
	echo
done

mv analysis-manufactured/params.bak common-manufactured/params.txt

cd $DATADIR

mv ../gpu-cuda-manufactured/runlog.csv temporal-convergence.csv

python3 mms-analysis.py
