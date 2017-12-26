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

# This script will build and execute diffusion programs for CPU and GPU architectures
# on a square mesh with edge length L=256 units, using the following set of mesh
# resolutions: 256, 512, 768, 1024, 1280, 1536, 1792, & 2048 points per dimension.
# The results are stored in the separate program directories, moved from `runlog.csv`
# to `runlog_<edge>.csv. These are then used by plot_runtimes.py to create scaling
# plots of runtime as a function of domain size (L×L) for each successful simulation.

DATADIR=$(pwd)
find . -name "runlog*csv" -exec rm {} \;
cd `dirname "${DATADIR}"`
cp -a common-diffusion/params.txt analysis-diffusion/params.bak

for i in {1..8}
do
	NX=$(echo "print int(256*${i})" | python)
	DX=$(echo "print 1.0/${i}"      | python)
	DT=$(echo "print 0.1*${DX}*${DX}/(4.0*0.00625)" | python)
	NC=$(echo "print int(100000.01/${DT})"          | python)
	DC=$(echo "print int(${NC}/10)" | python)
	sed "s/NXY/${NX}/g;s/dXY/${DX}/g;s/NC/${NC}/g;s/dT/${DC}/g" analysis-diffusion/params.in > common-diffusion/params.txt
	echo "Running with domain size L=${NX}"
	make run
	for f in $(find . -name "runlog.csv")
	do
		tail -n 1 $f
		mv $f ${f/runlog./scaling_${NX}.}
	done
	sleep 1
	echo
done

mv analysis-diffusion/params.bak common-diffusion/params.txt

cd $DATADIR
python plot_runtimes.py

