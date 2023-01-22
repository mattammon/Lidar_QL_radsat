import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from siphon.radarserver import RadarServer
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from datetime import timedelta
import datetime as DT
import metpy.plots as mpplots
from netCDF4 import Dataset
import math 
import utils as func

from functions import closest_value_index as CVI

def quicklook_radsat(profiler, vad_file, dlfp_file, sat_path, time_interval, y_bounds, w_range, date, radar_site, save_loc):
    """
    This is the main program that produces the plots themselves. Import this function into a separate
    python file and run it with necessary inputs (explained below) to output plots in png format.
    
    Function Inputs:
    - profiler: this is simply the name of the profiler that the plot is relevant to (in string format)
                for titling purposes (i.e. "Lidar Truck", "CLAMPS 2")
    - vad_file: file path of dlvad NetCDF file relevant to the date and profiler for plotting (string format)
    - dlfp_file: file path of dlvad NetCDF file relevant to the desired date and profiler for plotting (string format)
    - sat_path: file path of NetCDF file containing satellite data (string format)
    - time_interval: time duration of vertical profile plots surrounding specific time (integer format)
    - y_bounds: height bounds for vertical profile plots. i.e. 0-3km plot would entail y_bounds = [0,3] (list format)
    - w_range: vmin and vmax values for vertical velocity plot (i.e. vmin = -10 m/s, vmax = 10 m/s ------> w_range = [-10,10]) (list format)
    - date: timestamp of desired date and time for plot in 'YYYYMMDDHHMM' string format
    - radar_site: 4 letter name of desired radar site in string format (i.e. Jackson, MS radar site is 'KDGX')
    - save_loc: file path of desired location for saving the figure (in string format). If saving is not desired input False
    
    *** Many functions are used within that are located and explained in the utils.py file ***
    """

    time,hgt,wspd,intensity,backscatter = func.dlfp_vars(dlfp_file)[0:5]
    time_vad, hgt_vad, wspd_vad, wdir, intensity_vad, lat, lon = func.dlvad_vars(vad_file)[0:7]
    loc = [lon[0], lat[0]]
    
    x_truck, y_truck = func.geodetic_to_geostat(loc[0],loc[1])
    lon_min, lat_min = func.geodetic_to_geostat(loc[0]-1,loc[1]-1)
    lon_max, lat_max = func.geodetic_to_geostat(loc[0]+1,loc[1]+1)
    bounds_sat = [lon_min ,lat_min, lon_max, lat_max]

    x_rad, y_rad, ref, ref_norm, ref_cmap, station_long, station_lat = func.radar_data(date, radar_site)
    x_M1, y_M1, rad_M1, x_M2, y_M2, rad_M2 = func.satellite_data(date, sat_path)
    
    time_format = "%Y%m%d%H%M"
    dt = datetime.strptime(date, time_format)
    hrmin = dt.hour + (dt.minute/60)
    x_bounds = [hrmin - ((time_interval/2)/60), hrmin + ((time_interval/2)/60)]
    time_bounds = np.where(np.logical_and(time >= x_bounds[0], time <= x_bounds[1]))

    U_sfc,V_sfc,U_500m,V_500m,U_1km,V_1km = func.barb_3lev(hrmin, time_vad, hgt_vad, wspd_vad, wdir, [0,0.5,1])
    
    fig = plt.figure(figsize=(30,30))
   
    vert = plt.subplot2grid((4, 4), (0, 0), rowspan = 1, colspan=2)
    bounds_vert = wspd[time_bounds]
    a = vert.pcolormesh(time[time_bounds], hgt, bounds_vert.T, vmin=w_range[0], vmax=w_range[1], cmap ='seismic', shading = 'auto')
    vert.set_title(f'{profiler} Vertical Velocity on {dt.month}-{dt.day}-{dt.year}', fontsize = 18)
    vert.set_ylabel('Height (km)')
    vert.xaxis.set_major_formatter(lambda x, pos: '{}'.format(str(DT.timedelta(hours=x))[0:8]))
    vert.set_ylim(y_bounds[0], y_bounds[1])
    plt.setp(vert.get_xticklabels(), rotation=30, ha="right")
    cb = plt.colorbar(a, ax = vert, shrink = 0.75, pad = 0.025)
    cb.ax.set_xlabel('m/s')
    
    backs = plt.subplot2grid((4, 4), (1, 0), rowspan = 1, colspan=2)
    bounds_backs = backscatter[time_bounds]
    b = backs.pcolormesh(time[time_bounds], hgt, np.log10(bounds_backs.T), vmin=-7, vmax=-3, cmap ='turbo', shading = 'auto')
    backs.set_title(f'{profiler} Backscatter on {dt.month}-{dt.day}-{dt.year}', fontsize = 18)
    backs.set_xlabel('Time (UTC)')
    backs.set_ylabel('Height (km)')
    backs.set_ylim(y_bounds[0], y_bounds[1])
    backs.xaxis.set_major_formatter(lambda x, pos: '{}'.format(str(DT.timedelta(hours=x))[0:8]))
    plt.setp(backs.get_xticklabels(), rotation=30, ha="right")
    cb2 = plt.colorbar(b, ax = backs, shrink = 0.75, pad = 0.025)
    cb2.ax.set_xlabel('1/ms')

    proj_radar = ccrs.LambertConformal(central_longitude=station_long, central_latitude=station_lat)
    radar = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2, projection = proj_radar)
    radar.add_feature(cfeature.COASTLINE, linewidth=2)
    radar.add_feature(cfeature.STATES)
    radar.set_extent([loc[0]-1, loc[0]+1, loc[1]-0.75, loc[1]+0.75])
    rad_mesh = radar.pcolormesh(x_rad, y_rad, ref, cmap=ref_cmap, norm=ref_norm, zorder=0)
    if dt.minute > 9:
        text = radar.text(0.5, 1.01, f'{radar_site} Reflectivity {dt.year}-{dt.month}-{dt.day}T{dt.hour}:{dt.minute}Z', transform=radar.transAxes,fontdict={'size':22},ha='center')
    else:
        text = radar.text(0.5, 1.01, f'{radar_site} Reflectivity {dt.year}-{dt.month}-{dt.day}T{dt.hour}:0{dt.minute}Z', transform=radar.transAxes,fontdict={'size':22},ha='center')
    truck_x, truck_y = func.geostat_to_geodetic(x_truck,y_truck)
    radar.scatter(0.5, 0.5, zorder = 2, marker = '*', s = 500, c = 'red',transform=radar.transAxes)

    satellite = func.base_map('subplot',bounds_sat, False)
    satellite.scatter(x_truck, y_truck, zorder = 2, marker = '*', s = 500, c = 'red')
    satellite.barbs(x_truck, y_truck, U_sfc, V_sfc, zorder = 2, length = 9, color = 'orange', linewidth = 2)
    satellite.barbs(x_truck, y_truck, U_500m, V_500m, zorder = 2, length = 9, color = 'red', linewidth = 2)
    satellite.barbs(x_truck, y_truck, U_1km, V_1km, zorder = 2, length = 9, color = 'purple', linewidth = 2)
    text3 = satellite.text(0, -0.05, 'Surface Wind', color = 'orange', transform=satellite.transAxes,fontdict={'size':22},ha='left')
    text4 = satellite.text(0.5, -0.05, '500m Wind', color = 'red', transform=satellite.transAxes,fontdict={'size':22},ha='center')
    text5 = satellite.text(1, -0.05, '1km Wind', color = 'purple', transform=satellite.transAxes,fontdict={'size':22},ha='right')
    satellite.pcolormesh(x_M1, y_M1, rad_M1, vmin = 0,vmax = 400, cmap='gist_gray', zorder=0)
    satellite.pcolormesh(x_M2, y_M2, rad_M2, vmin = 0,vmax = 400, cmap='gist_gray', zorder=0)
    if dt.minute > 9:
        text2 = satellite.text(0.5, 1.01, f'GOES-16 ABI CH02 Radiance {dt.year}-{dt.month}-{dt.day}T{dt.hour}:{dt.minute}Z', transform=satellite.transAxes,fontdict={'size':22},ha='center')
    else:
        text2 = satellite.text(0.5, 1.01, f'GOES-16 ABI CH02 Radiance {dt.year}-{dt.month}-{dt.day}T{dt.hour}:0{dt.minute}Z', transform=satellite.transAxes,fontdict={'size':22},ha='center')
        
    hodograph = func.hodograph(hrmin, time_vad, hgt_vad, wspd_vad, wdir, intensity_vad)
    if dt.minute > 9:
        hodograph.text(0.5, 1.01, f'Hodograph from {profiler} {dt.year}-{dt.month}-{dt.day}T{dt.hour}:{dt.minute}Z', transform=hodograph.transAxes,fontdict={'size':22},ha='center')
    else:
        hodograph.text(0.5, 1.01, f'Hodograph from {profiler} {dt.year}-{dt.month}-{dt.day}T{dt.hour}:0{dt.minute}Z', transform=hodograph.transAxes,fontdict={'size':22},ha='center')
    
    plt.tight_layout()
    
    if save_loc != 'none':
        if dt.minute > 9:
            plt.savefig('{}/{}_{}{}z.png'.format(save_loc,profiler,dt.hour,dt.minute, dpi = 300, facecolor = 'w'))
        else:
            plt.savefig('{}/{}_{}0{}z.png'.format(save_loc,profiler,dt.hour,dt.minute, dpi = 300, facecolor = 'w'))
    return fig

    
