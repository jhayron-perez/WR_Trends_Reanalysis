import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import glob
import xesmf as xe
import os
from multiprocessing import Pool


def regrid(ds, variable, reuse_weights=False,filename_weights = None):
    """
    Function to regrid onto coarser ERA5 grid (0.25-degree).
    Args:
        ds (xarray dataset): file.
        variable (str): variable.
        reuse_weights (boolean): Whether to use precomputed weights to speed up calculation.
                                 Defaults to ``False``.
        filename_weights (str): if reuse_weights is True, then a string for the weights path is needed.
    Returns:
        Regridded data file for use with machine learning model.
    """
    # ds.lon = ds.longitude
    # ds.lat = ds.latitude
    # ds = ds.rename({'latitude':'lat',
    #                'longitude':'lon'})
    ds_out = xe.util.grid_2d(lon0_b=0-0.5,   lon1_b=360-0.5, d_lon=1., 
                             lat0_b=-90-0.5, lat1_b=90,      d_lat=1.)
    
    if reuse_weights == False:
        regridder = xe.Regridder(ds, ds_out, method='nearest_s2d', reuse_weights=reuse_weights)
        return regridder(ds[variable]),regridder
    else:
        regridder = xe.Regridder(ds, ds_out, method='nearest_s2d', reuse_weights=reuse_weights,
            filename = filename_weights)
        return regridder(ds[variable])
    

def process_month(year_month):
    path_jra3q_data = "/glade/campaign/collections/rda/data/d640000/anl_p/"
    path_outputs = "/glade/derecho/scratch/jhayron/Z500_JRA3Q_Daily/"
    filename_weights = '/glade/u/home/jhayron/WR_Trends_Reanalysis/Scripts/regridder_z500.nc'
    
    reuse_weights = os.path.exists(filename_weights)
    
    year = year_month[0]
    month = year_month[1]
    
    directory = os.path.join(path_jra3q_data, f'{year}{month:02d}')
    temp_list_files = np.sort(glob.glob(directory+'/jra3q.anl_p.0_3_5.hgt-pres-an-gauss.*'))
    data_month = xr.concat([xr.open_dataset(temp_list_files[i]).sel(pressure_level=500)['hgt-pres-an-gauss'] for i in range(len(temp_list_files))], dim='time').to_dataset()
    data_daily = data_month.resample(time='1D').mean()
    for itime, time in enumerate(data_daily.time.values):
        dataset_temp = data_daily.isel(time = itime)
        # if (year==1948)&(month==1)&(itime==0):
        #     ds_1deg, regridder = regrid(dataset_temp, 'hgt-pres-an-gauss', False)
        #     regridder.to_netcdf('regridder_z500.nc')
        #     ds_1deg = ds_1deg.to_dataset(name='Z')
        #     ds_1deg['Z'].attrs['units'] = 'm'
        #     ds_1deg['Z'].attrs['long_name'] = 'Geopotential at 500hPa'
        # else:
        ds_1deg = regrid(dataset_temp, 'hgt-pres-an-gauss', True, 'regridder_z500.nc')
        ds_1deg = ds_1deg.to_dataset(name='Z')
        ds_1deg['Z'].attrs['units'] = 'm'
        ds_1deg['Z'].attrs['long_name'] = 'Geopotential at 500hPa'
        nameoutputfile = f"jraq_z500_{str(time)[:10].replace('-','_')}.nc"
        ds_1deg.to_netcdf(f'{path_outputs}{nameoutputfile}')
        
def main():
    # Prepare a list of file information tuples for multiprocessing
    file_info_list = []

    for year in range(1999, 2024):
        for month in range(1, 13):
            file_info_list.append((year, month))

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=20) as pool:
        pool.map(process_month, file_info_list)
        
if __name__ == "__main__":
    main()