#!/usr/bin/env python
"""Plot the time history of one or more variables stored in b2time.nc.

Reads ./b2time.nc (run it from the run directory, or from b2mn.exe.dir for a
running job) and plots each requested variable against the simulation time
'timesa' stored in the same file, one line per variable on a single axis.
Each variable is squeezed (size-1 dimensions dropped) and passed to plt.plot
against 'timesa', so its first remaining dimension must be time (e.g. the
time series nesepa, nesepm, nesepi). Arrays that still have more than 2
dimensions after squeezing have not been checked.

Usage (the variable names go in ONE quoted, space-separated argument):
    2dt.py "nesepa nesepm nesepi"

Needs netCDF4, numpy and matplotlib; opens a plot window (use ssh -X or a
local display). Any variable name that is not in b2time.nc aborts the script.

Original author: JDL
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

def main(plotvars):
    file_in = "b2time.nc"

    plotvars = plotvars.split()

    try:
        ncIn = Dataset(file_in)
    except:
        print("Error: Could not open "+file_in)
        exit(0)

    timesa = ncIn.variables['timesa'][:]

    myvar = {}
    for item in plotvars:
        try:
            myvar[item] = ncIn.variables[item][:]
        except:
            print("Error: Could not load variable "+item)
            exit(0)

#    print(np.shape(myvar[plotvars[0]]))
    plt.figure()
    for i,item in enumerate(plotvars):
        plt.plot(timesa,np.squeeze(myvar[plotvars[i]]),label=item,marker='x')

    plt.xlabel("time (s)")
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('plotvars', type = str,
                        help = 'Space-separated list of variables to plot from b2time.nc, '
                               'as a single quoted string, e.g. "nesepa nesepm nesepi"')
    args = parser.parse_args()

    main(args.plotvars)
