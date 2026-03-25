#!/usr/bin/env python3
"""

This script clips a specified rectangular area of interest from cum h5.

=====
Usage
=====
LiCSBAS_cum_clip.py -i cum.h5 -o cum_clipped.h5 [-r x1:x2/y1:y2] [-g lon1/lon2/lat1/lat2]

 -i  orig cum.h5
 -o  clipped output
 -r  Range to be clipped. Index starts from 0.
     0 for x2/y2 means all. (i.e., 0:0/0:0 means whole area).
 -g  Range to be clipped in geographical coordinates (deg).

"""
#%% Change log
'''
20260325 Milan Lazecky, Uni of Leeds
 - created from LiCSBAS05op_clip.py
'''

#%% Import
from LiCSBAS_meta import *
import LiCSBAS_tools_lib as tools_lib
import getopt
import os
import re
import sys
import glob
import shutil
import time
import numpy as np
import xarray as xr

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#%%
def main(argv=None):

    #%% Check argv
    if argv == None:
        argv = sys.argv

    start = time.time()
    #ver='1.14.1'; date=20230804; author="Yu Morishita and COMET dev team"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    #%% Set default
    inh5 = []
    outh5 = []
    range_str = []
    range_geo_str = []

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:r:g:", ["help"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                in_dir = a
            elif o == '-o':
                out_dir = a
            elif o == '-r':
                range_str = a
            elif o == '-g':
                range_geo_str = a

        if not inh5:
            raise Usage('No input file given, -i is not optional!')
        if not outh5:
            raise Usage('No output file given, -o is not optional!')
        if not range_str and not range_geo_str:
            raise Usage('No clip area given, use either-g or -r')
        if range_str and range_geo_str:
            raise Usage('Both -r and -g given, use either -r or -g not both!')

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2
    
    if not os.path.exists(inh5):
        print('ERROR: No '+inh5+' file exists')
        return 2
    
    # Reading the file
    a = xr.open_dataset(inh5)
    width, length = a['coh_avg'].shape
    postlat = float(a.data_vars['post_lat'].values)
    postlon = float(a.data_vars['post_lon'].values)
    lat1 = float(a.data_vars['corner_lat'].values)
    lon1 = float(a.data_vars['corner_lon'].values)
    #%% Check and set range to be clipped
    ### Read -r or -g option
    if range_str: ## -r
        if not tools_lib.read_range(range_str, width, length):
            print('\nERROR in {}\n'.format(range_str), file=sys.stderr)
            return 1
        else:
            x1, x2, y1, y2 = tools_lib.read_range(range_str, width, length)
    elif range_geo_str: ## -g
        if not tools_lib.read_range_geo(range_geo_str, width, length, lat1, postlat, lon1, postlon):
            print('\nERROR in {}\n'.format(range_geo_str), file=sys.stderr)
            return 1
        else:
            x1, x2, y1, y2 = tools_lib.read_range_geo(range_geo_str, width, length, lat1, postlat, lon1, postlon)
            range_str = '{}:{}/{}:{}'.format(x1, x2, y1, y2)
    
    #dtype = a.refarea.values.dtype  # .replace('159','333')
    # now.. it is weird but phony_dim_X differs in meaning! -- very fast ugly workaround (only assuming position 0 or 2+ for time):
    dim_l, dim_w = a.coh_avg.dims
    for dimc in a.cum.dims:
        if dimc not in [dim_l, dim_w]:
            dim_t = dimc
    # refering to the mean of the scene...
    if dim_t == 'phony_dim_0':
        b = a.sel(phony_dim_1=slice(y1, y2), phony_dim_2=slice(x1, x2))
        b['cum'].values = b['cum'].values - b['cum'].mean(dim=['phony_dim_1', 'phony_dim_2']).values[:, np.newaxis,
                                        np.newaxis]
    else:
        b = a.sel(phony_dim_0=slice(y1, y2), phony_dim_1=slice(x1, x2))
        b['cum'].values = b['cum'].values - b['cum'].mean(dim=['phony_dim_0', 'phony_dim_1']).values[:, np.newaxis,
                                        np.newaxis]
    b.vel.values = b.vel.values - b.vel.mean().values
    b.vintercept.values = b.vintercept.values - b.vintercept.mean().values
    b['refarea'].values=np.array('0:'+str(x2-x1)+'/0:'+str(y2-y1), dtype='<U15')
    cornerlon = np.min( [float(range_geo_str.split('/')[0]), float(range_geo_str.split('/')[1])] )
    cornerlat = np.max( [float(range_geo_str.split('/')[2]), float(range_geo_str.split('/')[3])] )
    # probably not, but there is a chance we need to do
    # cornerlon = cornerlon - postlon/2
    # cornerlat = cornerlat - postlat/2
    # again. most probably not... but didn't double check the tools_lib.read_range_geo etc.
    b['corner_lon'].values = np.array(cornerlon)
    b['corner_lat'].values = np.array(cornerlat)
    b.to_netcdf(outh5) #, engine="h5netcdf")


#%% main
if __name__ == "__main__":
    sys.exit(main())
