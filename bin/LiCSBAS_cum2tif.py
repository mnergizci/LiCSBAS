#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import os
import shutil
import glob
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_inv_lib as inv_lib
import LiCSBAS_plot_lib as plot_lib
"""
v1.0 20250731 Muhammet Nergizci, Uni of Leeds -Adapted from Dr. Yuan Gao and Dr. Pedro Espin Bedon


========
Overview
========
This script outputs a tif and png file of cumulative displacement from cum*.h5.

=====
Usage
=====
LiCSBAS_cum2tif_png.py -T TS_folder -i cum_h5 -m mask -p dem_par -f frame_number -d imd_s

  -T  Path to the folder. Default: TS_GEOCml10GACOSmask.
  -i  Path to the h5 file. Default: cum.h5.
  -m  Path to the mask file.
  -p  Path to the DEM parameter file. Default: EQA.dem_par
  -f  frame number of naming files (Default: current working directory name)
  -d  End date of cumulative displacement.
  --help  Show this help message and exit.

"""

def cum_tiff_png(TS_folder, cum_h5, mask, dem_par, frame_number, imd_s):
    cum_h5 = os.path.join(TS_folder, cum_h5)
    if not cum_h5.endswith('filt.h5'):
        cum_filt_h5 = 'cum_filt.h5'
        cum_filt_h5 = os.path.join(TS_folder, cum_filt_h5)
    GEOC_folder= TS_folder.split('_')[-1]
    dem_par = os.path.join(GEOC_folder, dem_par)
    mask = os.path.join(TS_folder,'results', mask)
    # breakpoint()
    os.system(f"LiCSBAS_cum2flt.py -i {cum_h5} -d {imd_s} -o {TS_folder}/results/cum_{imd_s}.mask --mask {mask}")
    os.system(f"LiCSBAS_flt2geotiff.py -i {TS_folder}/results/cum_{imd_s}.mask -p {dem_par} -o {frame_number}.cum_{imd_s}.mask.tif")
    os.system(f"LiCSBAS_disp_img.py -i {TS_folder}/results/cum_{imd_s}.mask -p {dem_par} --cmin -100 --cmax 100 --png {frame_number}.cum_{imd_s}.mask.png --title {frame_number}.cum_{imd_s}.mask")
    ##make_for cum.filt
    os.system(f"LiCSBAS_cum2flt.py -i {cum_filt_h5} -d {imd_s} -o {TS_folder}/results/cum_filt_{imd_s}.mask --mask {mask}")
    os.system(f"LiCSBAS_flt2geotiff.py -i {TS_folder}/results/cum_filt_{imd_s}.mask -p {dem_par} -o {frame_number}.cum_filt_{imd_s}.mask.tif")
    os.system(f"LiCSBAS_disp_img.py -i {TS_folder}/results/cum_filt_{imd_s}.mask -p {dem_par} --cmin -100 --cmax 100 --png {frame_number}.cum_filt_{imd_s}.mask.png --title {frame_number}.cum_filt_{imd_s}.mask")

    # Move the different format files including *.cum *.tif and *.png to their corresponding folders
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the cum.h5 file and output the binary, tif and png file of cumulative dispalcement of each epoch.")
    parser.add_argument("-T", dest="TS_folder", default="TS_GEOCml10GACOSmask", help="Path to the folder. Default: TS_GEOCml10GACOSmask.")
    parser.add_argument("-i", dest="cum_h5", default="cum.h5", help="Path to the h5 file. Default:cum.h5.")
    parser.add_argument("-m", dest="mask", default="mask", help="Path to the mask file.")
    parser.add_argument("-p", dest="dem_par", default="EQA.dem_par", help="Path to the DEM parameter file. Default=EQA.dem_par")
    parser.add_argument("-f", dest="frame_number", default=os.path.basename(os.getcwd()), help="frame number of naming files")
    parser.add_argument("-d", dest="imd_s", help="End date of cumulative displacement.")
    args = parser.parse_args()

    cum_tiff_png(args.TS_folder ,args.cum_h5, args.mask, args.dem_par, args.frame_number, args.imd_s)
    # cum_tiff_png(args.cum_h5, args.mask, args.dem_par)

