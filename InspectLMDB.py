# -*- coding: utf-8 -*-
"""
Simple Python script for reading images from a LMDB file using caffe's datum.
 Author: Dimitris Kastaniotis, PhD Candidate, University of Patras, Patras, Greece
 Lisence: Use it as you like
 Disclaimer: This script is provided without any guarantee of correctness- or that will fit your needs.
 Please use with care.
 
"""

import lmdb
import cv2
import re, fileinput, math
import numpy as np
import matplotlib.pyplot as plt
 
caffe_root = 'PATH_TO_CAFFE/python'  
import sys
sys.path.insert(0, caffe_root )
import caffe

from caffe.proto import caffe_pb2
 
 
# Check number of arguments:
if(len(sys.argv)<2):
    print 'Argument List:', str(sys.argv)
    print "Example Of Usage:"
    print sys.argv[0], "LMDBfolder/" 
    sys.exit()
 

DataBaseFolderPath= sys.argv[1]

#==========================================================
#          Open database and read data and labels (for debugging)
#==========================================================
 
lmdb_env = lmdb.open(DataBaseFolderPath)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()
for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    im = data.astype(np.float)
    im = (np.transpose(im, (1,2, 0))) # original (dim, col, row)
    print "label ", label
    plt.imshow(im)
    plt.show()
 
#===========================================================
#                       end of process
#===========================================================
sys.exit()



