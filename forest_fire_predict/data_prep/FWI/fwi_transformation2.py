import os
import xarray as xr
import numpy as np
import pyproj
from util.day_to_date import dayIndex_to_date
from shapely.geometry import Point
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed


def daily_fwi_file_transform(day_index, year, daily_data, savepath, old_crs, target_crs, bbox):

    # transform 1-365 into actual date
    daily_date = dayIndex_to_date(year, day_index)

    # get the data of the day and only take FWI
    # daily_data = fwi_ds.sel(Time=day_index)
    daily_attr = daily_data['FWI']
    daily_df = daily_attr.to_dataframe()  # upon the transformation, df has two rows (lat,lon) and two colmns (Time,target_attr)
    daily_df = daily_df.drop(columns='Time')

    # move all the index attr to columns and replace new rows with numeric index
    daily_df = daily_df.reset_index()

    filtered_df = daily_df[
        (daily_df['Longitude'] >= bbox[0]) &
        (daily_df['Longitude'] <= bbox[2]) &
        (daily_df['Latitude'] >= bbox[1]) &
        (daily_df['Latitude'] <= bbox[3])
        ]

    # generate gdf and only keep fwi and x and y
    geometry = [Point(xy) for xy in zip(filtered_df['Longitude'], filtered_df['Latitude'])]
    gdf = gpd.GeoDataFrame(filtered_df, geometry=geometry, crs=old_crs)  # Ensure the original CRS is correct
    gdf = gdf.drop(columns=['Longitude', 'Latitude'])

    # Now transform and save
    gdf_transformed = gdf.to_crs(target_crs)
    gdf_transformed.to_file(f'{savepath}/{str(daily_date)} fwi_EPSG3978.shp')


def fwi_transform():
    EPSG3978_Canada_Atlas_Lambert = pyproj.CRS("+proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs")
    from_path = '../../raw_data/weather'
    to_path = '../../raw_data/weather/fwi_EPSG3978'

    # left. bottom, right, top
    bbox = [218.994451, 41.669086, 307.383393, 83.116523]

    years = np.arange(1994, 2022)
    for year in years:
        print(f"start spliting and transforming year {str(year)}")

        yearly_save_path = f'{to_path}/{str(year)}'
        if not os.path.exists(yearly_save_path):
            os.makedirs(yearly_save_path)

        ### read the yearly fwi and split by days
        yearly_fwi_ds = xr.open_dataset(f'{from_path}/fire_weather_index_{str(year)}.nc')

        # np.arange() is not feasible for it cannot be recognized by timedelta and will cause error
        day_indexes = range(1, yearly_fwi_ds.coords['Time'].size + 1)

        old_crs = yearly_fwi_ds['crs'].attrs.get('proj4', None)
        if not old_crs:
            raise ValueError("PROJ4 string not found in 'crs' variable attributes")

        # multi tasking
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # create key-value (future-index) pair dictionary
            futures = { }

            for day_index in day_indexes:
                daily_data = yearly_fwi_ds.sel(Time=day_index)
                future = executor.submit(daily_fwi_file_transform, day_index, year, daily_data, yearly_save_path, old_crs,
                                         EPSG3978_Canada_Atlas_Lambert, bbox)
                futures[future] = day_index

            # Wait for all futures to complete
            for future in as_completed(futures):
                day = futures[future]
                try:
                    result = future.result()
                    print(f"{year}-{day} completed successfully.")
                except Exception as exc:
                    print(f"{year}-{day} generated an exception: {exc}")

        yearly_fwi_ds.close()


fwi_transform()