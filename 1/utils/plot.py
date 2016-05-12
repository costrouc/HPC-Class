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
    fields = 5
    
    header = reader.next()
    
    for row, record in enumerate(reader):
        if len(record) != fields:
            print "Skipping malformed record %s, contains %i fields (%i expected)" % (record, len(record), fields)
        else:
            records.append(record)

    fig, ax = plt.subplots(1, 1)
            
    ax.plot([row[xcol] for row in records], [row[ycol] for row in records], 'r-')
    fig.savefig(filename, dpi=300, transparent=True)

# Create Plots of the csv data from each run
plot_hw1_csv("data/norm.txt", "img/norm.png", 0, 4)
plot_hw1_csv("data/matvec.txt", "img/matvec.png", 0, 4)
plot_hw1_csv("data/matmat.txt", "img/matmat.png", 0, 4)

    
