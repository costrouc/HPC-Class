#!/bin/python

import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_hw1_csv(inputcsv, filename, xcol, ycol):

    reader = csv.reader(open(inputcsv, "rb"), 
                        delimiter='\t', quoting=csv.QUOTE_NONE)

    header = []
    records = []
    fields = 6
    
    header = reader.next()
    
    for row, record in enumerate(reader):
        if len(record) != fields:
            print "Skipping malformed record %s, contains %i fields (%i expected)" % (record, len(record), fields)
        else:
            records.append(record)

    fig, ax = plt.subplots(1, 1)

    
    mflops = []
    temp_mflops = 0.0
    for row in records:
        temp_mflops = temp_mflops + float(row[5])
        if int(row[0]) == 3:
            mflops.append(temp_mflops)
            temp_mflops = 0.0
        
    ax.plot(range(len(mflops)), mflops, 'r-')
    fig.savefig(filename, dpi=300, transparent=True)

from os import listdir
from os.path import isfile, join

datadir = "data/"
imgdir =  "img/"

datafiles = [ f.split(".")[0] for f in listdir(datadir) if isfile(join(datadir,f)) ]

for datafile in datafiles:
    plot_hw1_csv(datadir + datafile + ".txt", imgdir + datafile + ".png", 0, 5)


    
