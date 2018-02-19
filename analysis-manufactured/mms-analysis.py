#!/usr/bin/python
# coding: utf-8

#***********************************************************************************
# HiPerC: High Performance Computing Strategies for Boundary Value Problems
# written by Trevor Keller and available from https://github.com/usnistgov/hiperc
# This software was developed at the National Institute of Standards and Technology
# by employees of the Federal Government in the course of their official duties.
# Pursuant to title 17 section 105 of the United States Code this software is not
# subject to copyright protection and is in the public domain. NIST assumes no
# responsibility whatsoever for the use of this software by other parties, and makes
# no guarantees, expressed or implied, about its quality, reliability, or any other
# characteristic. We would appreciate acknowledgement if the software is used.
# This software can be redistributed and/or modified freely provided that any
# derivative works bear some notice that they are derived from it, and any modified
# versions bear some notice that they have been modified.
# Questions/comments to Trevor Keller (trevor.keller@nist.gov)
#***********************************************************************************

# Usage: python3 mms-analysis.py

import numpy as np
from sys import argv
from os import path
import matplotlib.pylab as plt

# Load data from convergence study
step,sim_time,cfl,dt,dx,L2,conv_time,step_time,IO_time,soln_time,run_time = np.loadtxt("spatial-convergence.csv", skiprows=1, delimiter=',', unpack=True)

# Convert to log-log space
x = np.log10(dx)
y = np.log10(L2)

# Perform linear regression on log-log data
fit = np.polyfit(x, y, 1)
fit_fn = np.poly1d(fit)

print("Fitting parameters: {0}".format(fit))

# Plot line of best fit, report slope
#plt.plot(dx, L2, 'yo', label="m={0}".format(fit[1]))
plt.plot(x, y, marker="o", color="black", label="m={0}".format(fit[1]))
plt.legend(loc='best')
plt.savefig("spatial-convergence.png", dpi=400, bbox_inches='tight')
