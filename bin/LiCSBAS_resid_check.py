#!/usr/bin/env python3
"""
========
Overview
========
This script checks the residual of LiCSBAS13 step for sbovl data to drop the highly noisy interferograms
===============
Input & output files
===============
Inputs in TS_GEOCml*/  :
 - info/
 -  - 13resid.txt
 Outputs in TS_GEOCml*/ :
 - info/
   - 11bad_ifg.txt    : List of updated bad ifgs 11 to rerun step13 discarded from further processing
=====
Usage
=====
LiCSBAS11_check_unw.py [-t tsadir]
"""
#%% Change log
'''
M. Nergizci, 2025-08-07
'''

#%% Import
from LiCSBAS_meta import *
import getopt
import os
import sys
import time
import shutil
import numpy as np
import datetime as dt
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
from scipy import stats

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg

#%% Helper
def days_between(pair):
    try:
        d1_str, d2_str = pair.split('_')
        d1 = datetime.strptime(d1_str, '%Y%m%d')
        d2 = datetime.strptime(d2_str, '%Y%m%d')
        return abs((d2 - d1).days)
    except:
        return -1

#%% Main
def main(argv=None):

    #%% Check argv
    if argv == None:
        argv = sys.argv

    start = time.time()
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    tsadir = []
 
    #%% Read options
  
    # Parse options
    try:
        opts, args = getopt.getopt(argv[1:], "ht:", ["help"])
    except getopt.GetoptError as err:
        print(str(err))
        print(__doc__)
        return 1

    for o, a in opts:
        if o in ("-h", "--help"):
            print(__doc__)
            return 0
        elif o == "-t":
            tsadir = a

    if not tsadir or not os.path.isdir(tsadir):
        print("ERROR: Please provide a valid TS directory using -t", flush=True)
        return 1
    
    infodir = os.path.join(tsadir, 'info')
    restxtfile = os.path.join(infodir,'13resid.txt')
    badifg_file = os.path.join(infodir, '11bad_ifg.txt')

    if not os.path.isfile(restxtfile):
        print("ERROR: Residual file {} does not exist!".format(restxtfile), flush=True)
        return 1
    
    pairs, rms_values = [], []
    with open(restxtfile, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append(parts[0])
                rms_values.append(float(parts[1]))
                
        
    pairs = np.array(pairs)
    rms_values = np.array(rms_values)

    # Zero residuals
    zero_mask = rms_values == 0.0
    zero_pairs = pairs[zero_mask]

    # Remove zeros for stats
    valid_mask = rms_values != 0.0
    pairs = pairs[valid_mask]
    rms_values = rms_values[valid_mask]

    # Thresholding
    mean = np.mean(rms_values)
    std = np.std(rms_values)
    threshold = mean + 2 * std

    outlier_indices = np.where(rms_values > threshold)[0]
    outlier_pairs = pairs[outlier_indices]

    # Combine all unique problematic pairs
    bad_set = set()
    bad_set.update(zero_pairs)
    bad_set.update(outlier_pairs)
    
    
    # # Sort and write output
    # bad_list = sorted(bad_set)
    # print(f"  Total bad interferograms: {len(bad_list)}")
    # if bad_list:
    #     with open(badifg_file, 'w') as f:
    #         for pair in bad_list:
    #             f.write(pair + '\n')
    #     print(f"  Written to: {badifg_file}")
    # else:
    #     print("  No bad interferograms found.")
    
    # Combine with existing if present
    existing_pairs = set()
    if os.path.exists(badifg_file):
        with open(badifg_file, 'r') as f:
            for line in f:
                if line.strip():
                    existing_pairs.add(line.strip())

    # Merge and sort all unique bad interferograms
    all_bad_pairs = existing_pairs.union(bad_set)
    bad_list = sorted(all_bad_pairs)

    # Write updated file
    with open(badifg_file, 'w') as f:
        for pair in bad_list:
            f.write(pair + '\n')
    
    
    return 0


#%% main
if __name__ == "__main__":
    sys.exit(main())

