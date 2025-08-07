#!/usr/bin/env python3

import argparse
import os
import h5py as h5

"""
v1.0 20250731 Muhammet Nergizci, Uni of Leeds -Adapted from Dr. Yuan Gao and Dr. Pedro Espin Bedon


========
Overview
========
This script outputs a tif and png file of cumulative displacement from cum*.h5.

=====
Usage
=====
LiCSBAS_cum2tif_png.py -T TS_folder -i cum_h5 -m mask -p dem_par -f frame_number -d imd_s -plate_motion

  -T  Path to the folder. Default: TS_GEOCml10GACOSmask.
  -i  Path to the h5 file. Default: cum.h5.
  -mask  Path to the mask file.
  -dem  Path to the DEM parameter file. Default: EQA.dem_par
  -f  frame number of naming files (Default: current working directory name)
  -p  primary date of cumulative displacement. Default is first date of the cum_filt.h5
  -s  secondary date of cumulative displacement.
  --help  Show this help message and exit.
  --plate_motion  If set, it will calculate the plate motion effect

"""

def cum_tiff_png(TS_folder, cum_h5, mask, dem_par, frame_number, imd_p, imd_s, plate_motion=False):
  # breakpoint()
  cum_name= cum_h5.split('.')[0]
  cum_h5 = os.path.join(TS_folder, cum_h5) 
  GEOC_folder = TS_folder.split('_')[-1]
  dem_par = os.path.join(GEOC_folder, dem_par)
  mask = os.path.join(TS_folder,'results', mask)
  
  #read cum for the default start
  cumh55 = h5.File(cum_h5,'r')
  imdates = cumh55['imdates'][()].astype(str).tolist()
  if not imd_p:
      imd_p = imdates[0]

  # breakpoint()
  os.system(f"LiCSBAS_cum2flt.py -i {cum_h5} -m {imd_p} -d {imd_s} -o {TS_folder}/results/{cum_name}_{imd_p}-{imd_s}.mask --mask {mask}")
  if plate_motion:
    print('Cumulated plate motion effect will be removed from the cumulative displacement.')
    os.system(f"LiCSBAS_cum_plate_motion.py -t {TS_folder} -f {frame_number} -p {imd_p} -s {imd_s} -o {frame_number}.{cum_name}_{imd_p}-{imd_s}.mask.eurasia.tif")
    os.system(f"LiCSBAS_disp_img.py -i {frame_number}.{cum_name}_{imd_p}-{imd_s}.mask.eurasia.tif -p {dem_par} --cmin -100 --cmax 100 --png {frame_number}.{cum_name}_{imd_p}-{imd_s}.mask.eurasia.png --title {frame_number}.{cum_name}_{imd_p}-{imd_s}.eurasia")
  else:
    os.system(f"LiCSBAS_flt2geotiff.py -i {TS_folder}/results/{cum_name}_{imd_p}-{imd_s}.mask -p {dem_par} -o {frame_number}.{cum_name}_{imd_p}-{imd_s}.mask.tif")
    os.system(f"LiCSBAS_disp_img.py -i {TS_folder}/results/{cum_name}_{imd_p}-{imd_s}.mask -p {dem_par} --cmin -100 --cmax 100 --png {frame_number}.{cum_name}_{imd_p}-{imd_s}.mask.png --title {frame_number}.{cum_name}_{imd_p}-{imd_s}.mask")
  # Move the different format files including *.cum *.tif and *.png to their corresponding folders
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the cum.h5 file and output the binary, tif and png file of cumulative dispalcement of each epoch.")
    parser.add_argument("-T", dest="TS_folder", default="TS_GEOCml10GACOSmask", help="Path to the folder. Default: TS_GEOCml10GACOSmask.")
    parser.add_argument("-i", dest="cum_h5", default="cum_filt.h5", help="Path to the h5 file. Default:cum.h5.")
    parser.add_argument("-mask", dest="mask", default="mask", help="Path to the mask file.")
    parser.add_argument("-dem", dest="dem_par", default="EQA.dem_par", help="Path to the DEM parameter file. Default=EQA.dem_par")
    parser.add_argument("-f", dest="frame_number", default=os.path.basename(os.getcwd()), help="frame number of naming files")
    parser.add_argument("-p", dest="imd_p", default=None, help="Start date of cumulative displacement. Default is first date of the cum_filt.h5")
    parser.add_argument("-s", dest="imd_s", help="End date of cumulative displacement.")
    parser.add_argument("--plate_motion", action="store_true", help="If set, it will calculate the plate motion effect")
    args = parser.parse_args()

    cum_tiff_png(args.TS_folder ,args.cum_h5, args.mask, args.dem_par, args.frame_number, args.imd_p, args.imd_s, args.plate_motion)
    # cum_tiff_png(args.cum_h5, args.mask, args.dem_par)

