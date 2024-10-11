#!/usr/bin/env python3
"""
v1.1 20241011 Milan Lazecky, Leeds Uni

========
Overview
========
This script outputs a standard NetCDF4 file using LiCSBAS results


=====
Usage
=====
LiCSBAS_out2nc.py [-i infile] [-o outfile] [-m yyyymmdd]
     [--ref_geo lon1/lon2/lat1/lat2] [--clip_geo lon1/lon2/lat1/lat2] [-A]

 -i  Path to input cum file (Default: cum_filt.h5)
 -o  Output netCDF4 file (Default: output.nc)
 -m  Master (reference) date (Default: first date) - TODO: bperps are fixed-referred to the 1st date
 --alignsar, -A  Export complete cube as developed within AlignSAR (all amplitudes, coherences, calc D_A, mean amp, TODO: atmo_error based on step 16)
 --ref_geo  Reference area in geographical coordinates as: lon1/lon2/lat1/lat2
 --clip_geo  Area to clip in geographical coordinates as: lon1/lon2/lat1/lat2
 --compress, -C  use zlib compression (very small files but time series may take long to load in GIS)
 --postfilter will interpolate VEL only through empty areas and filter in space
 --apply_mask  Will apply mask to all relevant variables
 --extracol Will add extra layer from files in folder TS*/results - e.g. --extracol loop_ph_avg_abs
"""
#%% Change log
'''
v1.1 20241012 ML
 - allowing extras for AlignSAR cube
v1.05 20240420 ML
 - fixed masking (apply_mask), improved metadata (to be improved further)
v1.0 20200901 Milan Lazecky, Uni of Leeds
 - Original implementation
'''

#%% Import
import getopt
import os
import re
import sys
import time
import numpy as np
import datetime as dt
import xarray as xr
import rioxarray
import subprocess as subp
from scipy.ndimage import gaussian_filter
from scipy import interpolate

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg



def grep1line(arg,filename):
    file = open(filename, "r")
    res=''
    for line in file:
        if re.search(arg, line):
            res=line
            break
    file.close()
    if res:
        res = res.split('\n')[0]
    return res


#just an eye candy layer
def interp_and_smooth(da, sigma=0.8):
    dar = da.copy()
    array = np.ma.masked_invalid(dar.values)
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='linear')
    #, fill_value=0)
    GD1 = np.array(GD1)
    GD1 = gaussian_filter(GD1, sigma=sigma)
    dar.values = GD1
    return dar


def loadall2cube(cumfile, extracols=['loop_ph_avg_abs']):
    cumdir = os.path.dirname(cumfile)
    cohfile = os.path.join(cumdir,'results/coh_avg')
    rmsfile = os.path.join(cumdir,'results/resid_rms')
    vstdfile = os.path.join(cumdir,'results/vstd')
    stcfile = os.path.join(cumdir,'results/stc')
    maskfile = os.path.join(cumdir,'results/mask')
    metafile = os.path.join(cumdir,'../../metadata.txt')
    #h5datafile = 'cum.h5'
    cum = xr.load_dataset(cumfile)
    
    sizex = len(cum.vel[0])
    sizey = len(cum.vel)
    
    lon = cum.corner_lon.values+cum.post_lon.values*np.arange(sizex)-0.5*float(cum.post_lon)
    lat = cum.corner_lat.values+cum.post_lat.values*np.arange(sizey)+0.5*float(cum.post_lat)  # maybe needed? yes! for gridline/AREA that is default in rasterio...
    
    time = np.array(([dt.datetime.strptime(str(imd), '%Y%m%d') for imd in cum.imdates.values]))
    
    velxr = xr.DataArray(cum.vel.values.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
    #LiCSBAS uses 0 instead of nans...
    velxr = velxr.where(velxr!=0)
    velxr.attrs['unit'] = 'mm/year'
    #vinterceptxr = xr.DataArray(cum.vintercept.values.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
    
    cumxr = xr.DataArray(cum.cum.values, coords=[time, lat, lon], dims=["time","lat", "lon"])
    cumxr.attrs['unit'] = 'mm'
    #bperpxr = xr.DataArray(cum.bperp.values, coords=[time], dims=["time"])
    
    cube = xr.Dataset()
    cube['cum'] = cumxr
    cube['vel'] = velxr
    #cube['vintercept'] = vinterceptxr
    try:
        cube['bperp'] = xr.DataArray(cum.bperp.values, coords=[time], dims=["time"])
        cube['bperp'] = cube.bperp.where(cube.bperp!=0)
        # re-ref it to the first date
        if np.isnan(cube['bperp'][0]):
            firstbperp = 0
        else:
            firstbperp = cube['bperp'][0]
        cube['bperp'] = cube['bperp'] - firstbperp
        cube['bperp'] = cube.bperp.astype(np.float32)
        cube.bperp.attrs['unit'] = 'm'
    except:
        print('some error loading bperp info')
    
    #if 'mask' in cum:
    #    # means this is filtered version, i.e. cum_filt.h5
    cube.attrs['filtered_version'] = 'mask' in cum
    
    #add coh_avg resid_rms vstd
    if os.path.exists(cohfile):
        infile = np.fromfile(cohfile, 'float32')
        cohxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        cube['coh'] = cohxr
        cube.coh.attrs['unit']='unitless'
    else: print('No coh_avg file detected, skipping')
    if os.path.exists(rmsfile):
        infile = np.fromfile(rmsfile, 'float32')
        rmsxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        rmsxr.attrs['unit'] = 'mm'
        cube['rms'] = rmsxr
    else: print('No RMS file detected, skipping')
    try:
        for e in extracols:
            efile=os.path.join(cumdir,'results',e)
            if os.path.exists(efile):
                infile = np.fromfile(efile, 'float32')   # should be always float. but we can check with os.stat('loop_ph_avg_abs').st_size
                exr = xr.DataArray(infile.reshape(sizey, sizex), coords=[lat, lon], dims=["lat", "lon"])
                #rmsxr.attrs['unit'] = 'mm'
                cube[e] = exr
            else:
                print('No '+e+' file detected, skipping')
    except:
        print('debug - extra layers not included')
    if os.path.exists(vstdfile):
        infile = np.fromfile(vstdfile, 'float32')
        vstdxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        vstdxr.attrs['unit'] = 'mm/year'
        cube['vstd'] = vstdxr
    else: print('No vstd file detected, skipping')
    if os.path.exists(stcfile):
        infile = np.fromfile(stcfile, 'float32')
        stcxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        stcxr.attrs['unit'] = 'mm'
        cube['stc'] = stcxr
    else: print('No stc file detected, skipping')
    if os.path.exists(maskfile):
        infile = np.fromfile(maskfile, 'float32')
        #infile = np.nan_to_num(infile,0).astype(int)  # change nans to 0
        infile = np.nan_to_num(infile,0).astype(np.int8)  # change nans to 0
        maskxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        maskxr.attrs['unit'] = 'unitless'
        cube['mask'] = maskxr
    else: print('No mask file detected, skipping')
    # add inc_angle
    if os.path.exists(metafile):
        #a = subp.run(['grep','inc_angle', metafile], stdout=subp.PIPE)
        #inc_angle = float(a.stdout.decode('utf-8').split('=')[1])
        inc_angle = float(grep1line('inc_angle',metafile).split('=')[1])
        cube.attrs['inc_angle'] = inc_angle
    else: print('')#'warning, metadata file not found. using general inc angle value')
        #inc_angle = 39
    
    #cube['bperp'] = bperpxr
    #cube[]
    cube.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    cube.rio.write_crs("EPSG:4326", inplace=True)
    cube = cube.sortby(['time','lon','lat'])
    return cube

#not in use now
def maskit(clipped, cohthres = 0.62, rmsthres = 5, vstdthres = 0.3):
    da = clipped.copy()
    out = da.where(clipped.coh>=cohthres) \
    .where(np.abs(clipped.rms)<=rmsthres) \
    .where(np.abs(clipped.vstd)<=vstdthres)
    return out


def toalignsar(tsdir, ncfile, outncfile):
    '''Will add some extras to the ncfile - need to have workdir with GEOC.MLI loaded - and for now, the multilook should be only ML = 1!'''
    docoh = True
    doamp = True
    cube=xr.open_dataset(ncfile) # only opening, not loading to memory
    workdir=os.path.dirname(tsdir)
    mldircode=tsdir.split('/')[-1].split('_')[-1]
    geocmldir=os.path.join(workdir, mldircode)
    try:
        ml = int(mldircode[6:].split('G')[0].split('m')[0].split('c')[0])
        if ml>1:
            print('WARNING, only ML1 will work ok for amplitudes (for now) - you used multilook '+str(ml)+'. Skipping amplitudes.')
            doamp = False
        else:
            doamp = True
    except:
        print('WARNING, could not identify ml factor from the folder name. Assuming not equal 1 == skipping amplitudes')
        doamp = False
    if not os.path.exists(geocmldir):
        docoh = False
    mlidir = os.path.join(workdir, 'GEOC.MLI')
    if not os.path.exists(mlidir):
        doamp = False
    #
    #
    #
    if doamp:
        #cube = xr.open_dataset(ncfile)
        var = cube['cum'] # will do 3D set
        new_var = xr.DataArray(data=np.zeros((var.shape)).astype(np.float32), dims=var.dims)
        cube = cube.assign({'amplitude': new_var})
        print('Importing amplitudes')
        cube = import_tifs2cube_simple(mlidir, cube, searchstring='/*/*geo.mli.tif', varname='amplitude', thirddim='time',
                                apply_func=np.sqrt)
        cube['amplitude']=cube['amplitude'].where(cube['amplitude']!=0)
        print('calculating mean amp and amp stab index')
        cube['amp_mean']=cube.amplitude.mean(dim='time')
        cube['amp_std']=cube.amplitude.std(dim='time')
        #cube['ADI']=(cube.amp_std**2)/cube['amp_mean']
        cube['ampstab'] = 1 - cube['amp_mean'] / (cube.amp_std ** 2)  # from 0-1, close to 0 = very stable
        cube['ampstab'].values[cube['ampstab'] <= 0] = 0.00001
        cube.to_netcdf(outncfile) # uncompressed
    #if docoh:
    #    #
    #cube.to_netcdf(outnc, mode='w', unlimited_dims=['time'])
    #del cube # clean memory
    return cube

"""
os.chdir(os.environ['BATCH_CACHE_DIR'])
def import_mergedcohs2nc(instr = 'mergedcoh.btemp_', outncfile = 'cohcube.almostdone.nc'):
    filenames = glob.glob(instr+'*.tif')
    avg_coh = ''
    for infile in filenames:
        numdays = int(infile.replace(instr,'').split('.')[0])
        cc = rioxarray.open_rasterio(infile)
        cc = cc.rename({'band':'btemp'})
        cc['btemp'] = [numdays]
        if cc.max()<2:
            print('converting to int')
            cc = (cc*255).astype(np.uint8)
        if not isinstance(avg_coh, xr.DataArray):
            avg_coh = cc.copy(deep=True)
        else:
            avg_coh= xr.concat([avg_coh, cc], dim='btemp')
    cohcube = xr.Dataset()
    cohcube['avg_coh'] = avg_coh.astype(np.uint8) # just in case
    cohcube = cohcube.sortby('btemp')
    encode = {'avg_coh': {'zlib': True, 'complevel': 9}}
    coordsys = "epsg:4326"
    cohcube = cohcube.rio.write_crs(coordsys, inplace=True)
    if outncfile:
        if os.path.exists(outncfile):
            os.remove(outncfile)
            try:
                os.remove(outncfile+'.nocompressed.nc')
            except:
                print('')
        #cohcube.to_netcdf(outncfile+'.nocompressed.nc')
        cohcube.to_netcdf(outncfile, encoding=encode)
    return cohcube

"""

def import_tifs2cube_simple(tifspath, cube, searchstring='/*/*geo.mli.tif', varname = 'amplitude', thirddim = 'time', apply_func = None):
    '''e.g. for amplitude from mlis, use apply_func = np.sqrt
    Note this function is simplified and loads everything to memory! see licsar_extra/lics_tstools for improved way
    finally, the varname must exist!'''
    import glob
    import pandas as pd
    t=cube.indexes['time']
    tifs=glob.glob(tifspath+searchstring)
    for tif in tifs:
        fname = os.path.basename(tif)
        epoch=fname.split('.')[0]
        if '_' in epoch:  # in case of ifgs, we set this to the later date
            epoch = epoch.split('_')[-1]
        epochdt = pd.Timestamp(epoch)
        if not epochdt in t:
            continue
        i = t.get_loc(epochdt)
        try:
            data = rioxarray.open_rasterio(tif)
        except:
            print('ERROR loading tif for epoch '+epoch)
            continue
        data = np.flipud(data.values[0])
        if apply_func:
            data = apply_func(data)
        cube[varname].isel(time=i)[:] = data
    return cube


#%% Main
def main(argv=None):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    ver=1.1; date=20241011; author="M.Lazecky"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    cumfile = 'cum.h5'
    outfile = 'output.nc'
    imd_m = []
    #refarea = []
    refarea_geo = []
    #maskfile = []
    apply_mask = False
    cliparea_geo = []
    compress = False
    postfilter = False
    alignsar = False
    centre_refx, centre_refy = np.nan, np.nan
    extracols = ['loop_ph_avg_abs']

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:m:r:CA", ["help", "alignsar", "extracol=", "compress","postfilter","clip_geo=", "ref_geo=", "apply_mask", "mask="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                cumfile = a
            elif o == '-o':
                outfile = a
            elif o == '--extracol':
                extracols.append(a)
            elif o == '-m':
                imd_m = a
            elif (o == '-C') or (o=='--compress'):
                compress = True
                print('will use zlib compression')
            elif (o == 'postfilter'):
                postfilter = True
                print('vel_filt will be created including interpolation over masked area')
            elif o == '-r':
                refarea = a
                print('ref area in radar coords not implemented yet')
            elif o == '--clip_geo':
                cliparea_geo = a
                minclipx, maxclipx, minclipy, maxclipy = cliparea_geo.split('/')
                minclipx, maxclipx, minclipy, maxclipy = float(minclipx), float(maxclipx), float(minclipy), float(maxclipy)
            elif o == '--ref_geo':
                refarea_geo = a
                minrefx, maxrefx, minrefy, maxrefy = refarea_geo.split('/')
                minrefx, maxrefx, minrefy, maxrefy = float(minrefx), float(maxrefx), float(minrefy), float(maxrefy)
                centre_refx, centre_refy = (minrefx+maxrefx)/2, (minrefy+maxrefy)/2
            #elif o == '--mask':
            #    maskfile = a
            elif o == '--apply_mask':
                apply_mask = True
            elif (o == '-A') or (o=='--alignsar'):
                alignsar = True
                print('outputting as the AlignSAR InSAR TS cube')

        if not os.path.exists(cumfile):
            raise Usage('No {} exists! Use -i option.'.format(cumfile))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

    cube = loadall2cube(cumfile, extracols = extracols)
    
    if apply_mask:
        davars = list(cube.data_vars)
        davars.remove('mask')
        for vbl in davars:
            if 'lat' in cube[vbl].coords:
                cube[vbl] = cube[vbl].where(cube.mask==1)
    
    #reference cum to time (first date will be 0)
    if not imd_m:
        imd_m = cube.time.isel(time=0).values.astype('str').split('T')[0]
    
    cube['cum'] = cube['cum'] - cube['cum'].sel(time=imd_m)
    
    #reference it
    if refarea_geo:
        #ref = cube.rio.clip_box(minrefx, minrefy, maxrefx, maxrefy)
        ref = cube.sel(lon=slice(minrefx, maxrefx), lat=slice(minrefy, maxrefy))
        if len(ref.vel) == 0:
            print('warning, no points in the reference area - will export without referencing')
        else:
            refcoh = ref.where(ref.coh >0.6)
            if refcoh.vel.count() < 2:
                print('warning, the ref area has low coherence! continuing anyway')
                refcoh = ref
            #for v in refcoh.data_vars.variables:
            #for v in ['cum', 'vel', 'vel_filt']:
            for v in ['cum', 'vel']:
                cube[v] = cube[v] - refcoh[v].median(["lat", "lon"])
    else:
        # just load default ref point
        #if np.isnan(centre_refx):
        if cube.attrs['filtered_version']:
            inref = '16ref'
        else:
            inref = '13ref'
        cumdir = os.path.dirname(cumfile)
        refkml = os.path.join(cumdir,'info',inref+'.kml')
        refcoords = grep1line('<coordinates>',refkml)
        refcoords = refcoords.split('>')[1].split('<')[0].split(',')
        centre_refx, centre_refy = float(refcoords[0]), float(refcoords[1])
    
    cube.attrs['ref_lon'] = centre_refx
    cube.attrs['ref_lat'] = centre_refy
    # netcdf does not support boolean, so:
    cube.attrs['filtered_version'] = cube.attrs['filtered_version']*1
    
    #only now will clip - this way the reference area can be outside the clip, if needed
    if cliparea_geo:
        cube = cube.sel(lon=slice(minclipx, maxclipx), lat=slice(minclipy, maxclipy))
    
    if postfilter:
        #do filtered (it is nice)
        cube['vel_filt'] = interp_and_smooth(cube['vel'], 0.5)
    #masked = maskit(clipped)
    #masked['vel_filt'] = clipped['vel_filt']
    #masked.to_netcdf(outfile)
    #just to make sure it is written..
    #check if it does not invert data!
    cube.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    cube.rio.write_crs("EPSG:4326", inplace=True)
    if compress:
        if postfilter:
            encode = {'cum': {'zlib': True, 'complevel': 9}, 'vel': {'zlib': True, 'complevel': 9},
            'coh': {'zlib': True, 'complevel': 9}, 'rms': {'zlib': True, 'complevel': 9},
            'stc': {'zlib': True, 'complevel': 9}, 'vel_filt': {'zlib': True, 'complevel': 9},
            'time': {'dtype': 'i4'}}
        else:
            encode = {'cum': {'zlib': True, 'complevel': 9}, 'vel': {'zlib': True, 'complevel': 9},
            'coh': {'zlib': True, 'complevel': 9}, 'rms': {'zlib': True, 'complevel': 9},
            'stc': {'zlib': True, 'complevel': 9},
            'time': {'dtype': 'i4'}}
        cube.to_netcdf(outfile, encoding=encode)
    else:
        cube.to_netcdf(outfile, encoding={'time': {'dtype': 'i4'}})
    if alignsar:
        '''
        alignsar:
- load cohs2cube, amplitude
- do mean ampl and amplitude dispersion
- add atmo error (LB16 edit)
- add land cover var
- add metadata to the cube
- add flowcharts to MUC - general and (updated) LiCSBAS
        '''
        # will just load it from stored since we will use the non-load approach for amps/cohs to save memory
        del cube
        cube = toalignsar(os.path.dirname(cumfile), outfile, outfile+'.tmp.nc')
    #
    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}\n'.format(outfile), flush=True)


#%% main
if __name__ == "__main__":
    sys.exit(main())
