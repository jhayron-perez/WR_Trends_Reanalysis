import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import glob
import xesmf as xe
from multiprocessing import Pool
import os

def regrid(ds, variable, reuse_weights=False, filename_weights=None):
    """
    Function to regrid onto coarser ERA5 grid (1-degree).
    Args:
        ds (xarray dataset): file.
        variable (str): variable.
        reuse_weights (boolean): Whether to use precomputed weights to speed up calculation.
                                 Defaults to ``False``.
        filename_weights (str): if reuse_weights is True, then a string for the weights path is needed.
    Returns:
        Regridded data file for use with machine learning model.
    """
    ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    ds_out = xe.util.grid_2d(lon0_b=0-0.5, lon1_b=360-0.5, d_lon=1., 
                             lat0_b=-90-0.5, lat1_b=90, d_lat=1.)
    
    if not reuse_weights:
        regridder = xe.Regridder(ds, ds_out, method='nearest_s2d', reuse_weights=reuse_weights)
        return regridder(ds[variable]), regridder
    else:
        regridder = xe.Regridder(ds, ds_out, method='nearest_s2d', reuse_weights=reuse_weights,
                                 filename=filename_weights)
        return regridder(ds[variable])

def process_file(file_info):
    """
    Process a single file for a given year and month.
    Args:
        file_info (tuple): Contains year, month, path_era5_data, path_outputs, and file.
    """
    year, month, file, path_outputs, reuse_weights, filename_weights = file_info
    
    # Open the dataset
    dataset_temp = xr.open_dataset(file)
    # Select data at the 500 hPa level
    dataset_temp = dataset_temp.sel(level=500)
    
    # Determine if it's the first file to create the regridder
    if reuse_weights == False:
        ds_1deg, regridder = regrid(dataset_temp, 'Z', False)
        # Save the regridder weights for future use
        regridder.to_netcdf(filename_weights)
    else:
        ds_1deg = regrid(dataset_temp, 'Z', True, filename_weights)
    
    # Convert to dataset and set attributes
    ds_1deg = ds_1deg.to_dataset(name='Z')
    ds_1deg['Z'].attrs['units'] = 'm**2 s**-2'
    ds_1deg['Z'].attrs['long_name'] = 'Geopotential at 500hPa'
    
    # Calculate the daily mean
    ds_1deg_daily = ds_1deg.mean('time')
    
    # Create output file name
    nameoutputfile = f"era5_z500_{str(dataset_temp.time.values[0])[:10].replace('-','_')}.nc"
    # Save to NetCDF
    ds_1deg_daily.to_netcdf(f'{path_outputs}{nameoutputfile}')
    
    print(f"Processed {nameoutputfile}")

def main():
    path_era5_data = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/"
    path_outputs = "/glade/derecho/scratch/jhayron/Z500_ERA5_Daily/"
    filename_weights = '/glade/u/home/jhayron/WR_ClimateChange/Scripts/regridder_z500.nc'
    
    reuse_weights = os.path.exists(filename_weights)
    
    # Prepare a list of file information tuples for multiprocessing
    file_info_list = []

    for year in range(1992, 2024):
        for month in range(1, 13):
            temp_list_files = np.sort(glob.glob(f"{path_era5_data}{year}{str(month).zfill(2)}/e5.oper.an.pl.128_129_z.ll025sc.*.nc"))
            
            for file in temp_list_files:
                file_info_list.append((year, month, file, path_outputs, reuse_weights, filename_weights))

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=72) as pool:
        pool.map(process_file, file_info_list)
        
if __name__ == "__main__":
    main()



# path_data = "/glade/derecho/scratch/jhayron/Z500_ERA5_Daily/"
# files_era5 = np.sort(glob.glob(f"{path_data}*.nc"))
# datasets = [xr.open_dataset(f) for f in files_era5]

# # Concatenate along a new 'time' dimension
# combined_dataset = xr.concat(datasets, dim='time')
# dataset = xr.open_dataset('/glade/derecho/scratch/jhayron/Z500_ERA5_Daily.nc')
# dates = [np.datetime64('1940-01-01') + np.timedelta64(i, 'D') for i in range(len(dataset.time))]
# dataset = dataset.assign_coords(time=("time", dates))
# dataset.to_netcdf('/glade/derecho/scratch/jhayron/Data4WRsClimateChange/ProcessedDataReanalyses/Z500_ERA5.nc')