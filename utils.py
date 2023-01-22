import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from siphon.radarserver import RadarServer
import cartopy
import cartopy.crs as ccrs
from datetime import datetime
from datetime import timedelta
import metpy.plots as mpplots
from netCDF4 import Dataset
import math 
import shutil
import tarfile
import os

def closest_value_index(variable, desired_value):
    """
    This function is a simple utility that returns the index of a desired value in any array/variable
    """
    
    arr = np.asarray(variable)
    location = (np.abs(arr - desired_value)).argmin()
    return location

def wind_comps_2(wspd, wdir):
    """
    Returns arrays of u and v wind components (in knots) using a wind direction array (wdir) 
    and wind speed array (wspd) as inputs
    
    note: The component derivation assumes the input wdir values are in degrees corresponding
          to the unit circle i.e. 90 degrees in north, 180 is south, etc. and outputs components
          that are in respect to the wind directions: N: 0, E: 90, S: 180, W: 270
    """
    wdir = np.radians(wdir)
    wspd = wspd*1.944
    u = -wspd*np.sin(wdir)
    v = -wspd*np.cos(wdir)
    return u, v

def polar_to_cartesian(az, rng):
    """
    Converts polar coordinates from radar data to cartesian for plotting purposes
    """
    
    az_rad = np.deg2rad(az)[:, None]
    x = rng * np.sin(az_rad)
    y = rng * np.cos(az_rad)
    return x, y

def dlfp_vars(file):
    """
    Returns arrays of dlfp variables from input file path that are relevant to plots 
    """
    
    nc = Dataset(file)
    time = np.array(nc.variables['hour'][:])
    height =  np.array(nc.variables['height'][:])
    velocity =  np.array(nc.variables['velocity'][:])
    intensity =  np.array(nc.variables['intensity'][:])
    azimuth =  np.array(nc.variables['azimuth'][:])
    elevation =  np.array(nc.variables['elevation'][:])
    pitch =  np.array(nc.variables['pitch'][:])
    cbh =  np.array(nc.variables['cbh'][:])
    backscatter =  np.array(nc.variables['backscatter'][:])
    lat =  np.array(nc.variables['lat'][:])
    lon =  np.array(nc.variables['lon'][:])
    return time, height, velocity, intensity, backscatter, azimuth, elevation, pitch, cbh, lat, lon

def dlvad_vars(file):
    """
    Returns arrays of vad variables from input file path that are relevant to plots 
    """
        
    nc = Dataset(file)
    time = nc.variables['hour'][:]
    height = nc.variables['height'][:]
    intensity = nc.variables['intensity'][:]
    wspd = nc.variables['wspd'][:]
    wdir = nc.variables['wdir'][:]
    w = nc.variables['w'][:]
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    return time, height, wspd, wdir, intensity, lat, lon, w

def base_map(typ, bounds, grid):
    """
    Returns ax object of map onto which satellite data will be plotted using 
    geostationary coordinate reference system
    """
    
    proj = ccrs.Geostationary(central_longitude = -75, satellite_height=35786023)
    
    if typ == 'subplot': 
        ax = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2, projection = proj)
    else:
        ax = plt.subplot(projection = proj)
    if grid == True:
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
    ax.add_feature(cartopy.feature.OCEAN) 
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_xlim(bounds[0],bounds[2])
    ax.set_ylim(bounds[1],bounds[3])
    return ax

def geostat_to_geodetic(x,y):
    """
    This function converts an array of geostationary (x,y) coordinates to corresponding
    arrays of geodetic coordinates (longitude, latitude)
    """

    a = np.sin(x)**2 + ((np.cos(x)**2)*(np.cos(y)**2 + ((6378137**2)/(6356752.31414**2))*(np.sin(y)**2)))
    b = -2*42164160*np.cos(x)*np.cos(y)
    c = (42164160**2) - (6378137**2)
    rs = (-b - np.sqrt((b**2)-4*a*c))/(2*a)
    sx = rs*np.cos(x)*np.cos(y)
    sy = -rs*np.sin(x)
    sz = rs*np.cos(x)*np.sin(y)
    
    lats = np.arctan(((6378137**2)/(6356752.31414**2))*(sz/(np.sqrt(((42164160 - sx)**2) + sy**2))))
    longs = -1.308996939 - np.arctan(sy/(42164160-sx))
    lats = np.degrees(lats)
    longs = np.degrees(longs)
    return longs, lats


def geodetic_to_geostat(longs,lats):
    """
    This function converts an array of geodetic coordinates (longitude, latitude) to 
    corresponding arrays of geostationary coordinates (x,y)
    
    note: This function assumes that the altitude of the satellite is 35,786,023 meters
          which is valid for the GOES-16 satellite
    """
        
    lats = np.radians(lats)
    longs = np.radians(longs)
    
    H = 42164160
    e = 0.0818191910435
    r_pol = 6356752.31414
    r_eq = 6378137
    lon_0 = -1.308996939
    lat_c = np.arctan(((r_pol**2)/(r_eq**2))*np.tan(lats))
    rc = r_pol/np.sqrt(1 - (e**2)*(np.cos(lat_c)**2))
    sx = H - (((rc)*np.cos(lat_c))*(np.cos(longs - lon_0)))
    sy = -((rc)*np.cos(lat_c))*(np.sin(longs - lon_0))
    sz = (rc)*np.sin(lat_c)
    
    y = np.arctan(sz/sx)
    x = np.arcsin(-sy/np.sqrt((sx**2)+(sy**2)+(sz**2)))
    return 35786023*x,35786023*y

def radar_data(stamp, site):
    """
    Inputs: timestamp string in YYYYMMDDHHMM format
            radar site in string format (i.e. Memphis Radar: 'KNQA', Jackson Radar: 'KDGX')
            
    Outputs: x,y : cartesian coordinate arrays
             ref: array of reflectivity values 
             ref_norm: Normalization of reflectivity values (applied during plotting)
             ref_cmap: Reflectivity colormap
             Longitude and Latitude of radar station
    """
    
    rs = RadarServer('http://tds-nexrad.scigw.unidata.ucar.edu/thredds/radarServer/nexrad/level2/S3/')
    query = rs.query()
    time_format = "%Y%m%d%H%M"
    dt = datetime.strptime(stamp, time_format)
    query.stations(site).time_range(dt, dt + timedelta(hours=1))
    cat = rs.get_catalog(query)
    cat.datasets
    ds = cat.datasets[0]
    data = ds.remote_access(service = 'OPENDAP')
    sweep = 0
    
    ref_var = data.variables['Reflectivity_HI']
    ref_data = np.array(ref_var[sweep])
    rng = data.variables['distanceR_HI'][:]
    az = data.variables['azimuthR_HI'][sweep]
    ref = np.ma.array(ref_var[sweep], mask=ref_var[sweep]==0)
    
    x, y = polar_to_cartesian(az, rng)
    ref_norm, ref_cmap = mpplots.ctables.registry.get_with_steps('NWSReflectivity', 5, 5)
    return x, y, ref, ref_norm, ref_cmap, data.StationLongitude, data.StationLatitude

def tarfile_extract(tar_path, cdf_path):
    """
    Satellite data ordered from EOL comes in tar.gz format. This function simply extracts
    all netCDF files from a tar.gz file at tar_path to any folder located at cdf_path
    """
    
    tar = tarfile.open(tar_path, "r:gz")
    tar.extractall(cdf_path)
    
def rename_sat_file(path, new_folder):
    """
    The given file names of the extracted netCDF files are long and complicated making them
    difficult to access for a given date and time. This function renames these files (singular
    files located at "path" input variable) to much shorter file names that are easier to reference 
    in following functions that pull these files to access their data
    
    Additionally, the extracted netCDF files include tons of different satellite data from ~16 different
    channels. For the pupose of these plots, we are only concerned with channel 2 0.5km resolution
    visible satellite data. This function gives the option to pull only these relevant files and move
    them to a different folder if desired. The "new_folder" input variable should be the path to the 
    folder that these files should be moved to. If this step is not necessary or not desired for your 
    personal use, simply put 'none' in string format for this variable
    """
    
    dir_list = os.listdir(path)
    for file in dir_list:
        if 'RadM1-M6C02' in file:
            date = file[28:39]
            new = f'{path}/RadM1-M6C02_{date}.nc'
            os.rename('{}/{}'.format(path,file), new)
            if new_folder != False:
                shutil.move(new, new_folder)
                
        if 'RadM2-M6C02' in file:
            date = file[28:39]
            new = f'{path}/RadM2-M6C02_{date}.nc'
            os.rename('{}/{}'.format(path,file), new)
            if new_folder != False:
                shutil.move(new, new_folder)
            
def satellite_data(stamp, path):
    """
    This function accesses visible satellite data from file located at "path" input, at desired
    date and time denoted by "stamp" variable which should be in YYYYMMDDHHMM string format. The 
    outputs include x and y geostationary location arrays for the satellite data as well as radiance
    values at the given coordinate array locations
    
    note: the satellite data for a given time is extracted from 2 separate nc files. This is because
    data is separated by "meso-sectors" that encompass different regions. In order to include the entire
    southeastern United States, it is necessary to include data from both meso-sector 1 and meso-sector 2.
    The meso-sector is denoted in the file name by RadM# where # is 1 for meso-sector 1 and 2 for meso-secor 2.
    i.e. RadM1-M6C02_20220812000.nc is data for meso-sector 1 at 2000z on 3/22/2022 while
         RadM2-M6C02_20220812000.nc is data for meso-sector 2 at 2000z on 3/22/2022
         
    ***IMPORTANT***: This function only works properly if the files are renamed according to the above
                     rename_sat_file() function
    """
    
    time_format = "%Y%m%d%H%M"
    dt = datetime.strptime(stamp, time_format)
    
    ## The file naming format references the day of the year in DDD format (doy variable) so 3/22/22 is the
    #  81st day of the year so doy is 081 while 4/13/22 is the 103rd day of the year so doy is 103. doy is calculated 
    #  automatically below.
    
    doy = (dt - datetime(dt.year,1,1))
    doy = str(doy)[0:3]
    doy = str(int(doy)+1)

    if int(doy) > 99:
        if dt.minute > 9:
            sat_1 = Dataset(r'{}/RadM1-M6C02_{}{}{}{}.nc'.format(path,dt.year,doy,dt.hour,dt.minute))
        else:
            sat_1 = Dataset(r'{}/RadM1-M6C02_{}{}{}0{}.nc'.format(path,dt.year,doy,dt.hour,dt.minute))
        if dt.minute > 9:
            sat_2 = Dataset(r'{}/RadM2-M6C02_{}{}{}{}.nc'.format(path,dt.year,doy,dt.hour,dt.minute))
        else:
            sat_2 = Dataset(r'{}/RadM2-M6C02_{}{}{}0{}.nc'.format(path,dt.year,doy,dt.hour,dt.minute))
    else:
        if dt.minute > 9:
            sat_1 = Dataset(r'{}/RadM1-M6C02_{}0{}{}{}.nc'.format(path,dt.year,doy,dt.hour,dt.minute))
        else:
            sat_1 = Dataset(r'{}/RadM1-M6C02_{}0{}{}0{}.nc'.format(path,dt.year,doy,dt.hour,dt.minute))
        if dt.minute > 9:
            sat_2 = Dataset(r'{}/RadM2-M6C02_{}0{}{}{}.nc'.format(path,dt.year,doy,dt.hour,dt.minute))
        else:
            sat_2 = Dataset(r'{}/RadM2-M6C02_{}0{}{}0{}.nc'.format(path,dt.year,doy,dt.hour,dt.minute))
    
    radiance_1 = sat_1.variables['Rad'][:]
    x_1 = sat_1.variables['x'][:]
    y_1 = sat_1.variables['y'][:]
    
    radiance_2 = sat_2.variables['Rad'][:]
    x_2 = sat_2.variables['x'][:]
    y_2 = sat_2.variables['y'][:]
    return 35786023*x_1, 35786023*y_1, radiance_1, 35786023*x_2, 35786023*y_2, radiance_2

def barb_3lev(time, time_vad, hgt, wspd, wdir, levels):
    """
    This function returns u and v wind components at 3 different levels (the "levels" input variable
    which should be in list format such as [level1, level2, level3] where the levels are integers in km AGL)
    to be plotted as windbarbs at the lidar location.
    
    Inputs: time: desired time of barbs in HHMM format (i.e. 2030z)
            time_vad: time array from relevant dlvad file
            hgt: height array from relevant dlvad file
            wspd: wspd array from relevant dlvad file
            wdir: wdir array from relevant dlvad file
            levels: described above
            
    Note: the u and v values are computed by taking the average of the mean and median of the 5 values surrounding
          and including the desired level to minimize noise and impact of outlier values.
    """
    
    time_barb = closest_value_index(time_vad, time)
    
    if levels[0] == 0:
        lev_1 = [2,7]
    else:
        lev_1 = [closest_value_index(hgt,levels[0])-3,closest_value_index(hgt,levels[0])+3]
        
    speed_L1 = (np.nanmean(wspd[time_barb,lev_1[0]:lev_1[1]]) + np.nanmedian(wspd[time_barb,lev_1[0]:lev_1[1]]))/2
    U_1,V_1 = wind_comps_2(speed_L1,wdir[time_barb,lev_1[0]:lev_1[1]])
    U_1,V_1 = np.nanmean(U_1), np.nanmean(V_1)
    
    lev_2 = [closest_value_index(hgt,levels[1])-3,closest_value_index(hgt,levels[1])+3]
    speed_L2 = (np.nanmean(wspd[time_barb,lev_2[0]:lev_2[1]]) + np.nanmedian(wspd[time_barb,lev_2[0]:lev_2[1]]))/2
    U_2,V_2 = wind_comps_2(speed_L2,wdir[time_barb,lev_2[0]:lev_2[1]])
    U_2,V_2 = np.nanmean(U_2), np.nanmean(V_2)
    
    lev_3 = [closest_value_index(hgt,levels[2])-3,closest_value_index(hgt,levels[2])+3]
    speed_L3 = (np.nanmean(wspd[time_barb,lev_3[0]:lev_3[1]]) + np.nanmedian(wspd[time_barb,lev_3[0]:lev_3[1]]))/2
    U_3,V_3 = wind_comps_2(speed_L3,wdir[time_barb,lev_3[0]:lev_3[1]])
    U_3,V_3 = np.nanmean(U_3), np.nanmean(V_3)
    return U_1,V_1,U_2,V_2,U_3,V_3

def hodograph(time, time_vad, hgt, wspd, wdir, intensity):
    """
    This function outputs a hodograph axis object corresponding to the lidar-observed wind profile of the CBL.
    The values are computed by averaging every 2 levels of data that are filtered with an intensity threshold
    of 1.007 to reduce noise and eliminate unrelieable values.
    
    Inputs: time: desired time of hodograph in HHMM format (i.e. 2030z)
        time_vad: time array from relevant dlvad file
        hgt: height array from relevant dlvad file
        wspd: wspd array from relevant dlvad file
        wdir: wdir array from relevant dlvad file
        intensity: intensity array from relevant dlvad file
    """
    
    hgt = hgt[0:200:2]
    hodo_time = closest_value_index(time_vad, time)
    foo = np.where(intensity <= 1.007)
    wspd[foo], wdir[foo] = np.nan, np.nan
    u_arr, v_arr = wind_comps_2((wspd[hodo_time,:]), wdir[hodo_time,:])
    
    u_levs = np.array(np.nanmean(u_arr[0:2]))
    i = 2
    while i < 200:
        u_levs = np.append(u_levs, np.nanmean(u_arr[i:(i+2)]))
        i+=2
        
    v_levs = np.array(np.nanmean(v_arr[0:2]))
    i = 2
    while i < 200:
        v_levs = np.append(v_levs, np.nanmean(v_arr[i:(i+2)]))
        i+=2

    ax = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
    comp_max = [np.nanmax(u_arr), np.nanmax(v_arr)]
    max_range = np.sqrt(comp_max[0]**2 + comp_max[1]**2)
    hodo = mpplots.Hodograph(ax, component_range = math.ceil((max_range/10))*10)
    hodo_plot = hodo.plot_colormapped(u_levs, v_levs, hgt)
    hodo.add_grid(increment=20)
    cb = plt.colorbar(hodo_plot, ax = ax, shrink = 0.5, pad = 0.05)
    cb.ax.set_ylabel('km')
    return ax
    