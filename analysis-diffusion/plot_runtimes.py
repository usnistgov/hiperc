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

# Usage: python plot_runtimes.py

import glob
import numpy as np
from sys import argv
from os import path
import matplotlib.pylab as plt

cpuBase = ("serial", "openmp", "tbb")
gpuBase = ("cuda", "openacc", "opencl")

sizes = (256, 512, 768, 1024)

dirset = (["cpu-{0}-diffusion".format(c) for c in cpuBase],
          ["gpu-{0}-diffusion".format(g) for g in gpuBase])
dirs = [s for sublist in dirset for s in sublist]

colors = ['black'] + [plt.cm.cool(i) for i in np.linspace(0, 1, len(dirs)-1)]
markers = ('*', 'o', '^', 'p', 'H', '8', 'v', 'd')

plt.figure(0)
plt.title('Runtime')
plt.xlabel(r'Simulation Time')
plt.ylabel(r'Execution Time')

plt.figure(1)
plt.title('Residual')
plt.xlabel(r'Simulation Time')
plt.ylabel(r'Residual')

plt.figure(2)
plt.title('Convolution')
plt.xlabel(r'Simulation Time')
plt.ylabel(r'Convolution Time')

for nx in sizes:
    plt.figure(3)
    plt.title('Runtime with $N_x={0}$'.format(nx))
    plt.xlabel(r'Simulation Time')
    plt.ylabel(r'Execution Time')

    plt.figure(4)
    plt.title('Residual with $N_x={0}$'.format(nx))
    plt.xlabel(r'Simulation Time')
    plt.ylabel(r'Residual')

    plt.figure(5)
    plt.title('Diffusion with $N_x={0}$'.format(nx))
    plt.xlabel(r'Simulation Time')
    plt.ylabel(r'Convolution Time')

    plt.figure(6)
    plt.title('I/O with $N_x={0}$'.format(nx))
    plt.xlabel(r'Simulation Time')
    plt.ylabel(r'I/O Time')

    for j, dirname in enumerate(dirs):
        datdir = "../{0}".format(dirname)
        logfile = "{0}/scaling_{1}.csv".format(datdir, nx)
        if path.isdir(datdir) and len(glob.glob(logfile)) > 0:
            base = path.basename(datdir)
            step,sim_time,wrss,conv_time,step_time,IO_time,soln_time,run_time = np.loadtxt(logfile, skiprows=1, delimiter=',', unpack=True)

            plt.figure(0)
            plt.plot(sim_time, run_time, '-', color=colors[j], marker=markers[j])

            plt.figure(1)
            plt.plot(sim_time, wrss, '-', color=colors[j], marker=markers[j])

            plt.figure(2)
            plt.plot(sim_time, step_time, '-', color=colors[j], marker=markers[j])

            plt.figure(3)
            plt.plot(sim_time, run_time, '-', color=colors[j], marker=markers[j], label=dirs[j])

            plt.figure(4)
            plt.plot(sim_time, wrss, '-', color=colors[j], marker=markers[j], label=dirs[j])

            plt.figure(5)
            plt.plot(sim_time, step_time, '-', color=colors[j], marker=markers[j], label=dirs[j])

            plt.figure(6)
            plt.plot(sim_time, IO_time, '-', color=colors[j], marker=markers[j], label=dirs[j])
        else:
            print("Invalid argument: {0} is not a directory, or contains no usable data.".format(datdir))
            print("Usage: {0}".format(argv[0]))

    plt.figure(3)
    plt.legend(loc='best')
    plt.savefig("all-runtime-{0}.png".format(nx), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(4)
    plt.legend(loc='best')
    plt.savefig("all-residual-{0}.png".format(nx), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(5)
    plt.legend(loc='best')
    plt.savefig("all-diffusion-{0}.png".format(nx), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(6)
    plt.legend(loc='best')
    plt.savefig("output_{0}.png".format(nx), dpi=300, bbox_inches='tight')
    plt.close()

plt.figure(0)
plt.savefig("all-runtimes.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(1)
plt.savefig("all-residuals.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(2)
plt.savefig("all-diffusions.png", dpi=300, bbox_inches='tight')
plt.close()
