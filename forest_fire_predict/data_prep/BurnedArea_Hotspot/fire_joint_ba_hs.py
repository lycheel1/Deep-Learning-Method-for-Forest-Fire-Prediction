import os.path
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from util.date_list import generate_date_list
""" 1 Run first to generate fire events """

# The purpose of this file is to generate fires based on burned area and integrate information from hotspot
# for each fire, it goes through the hotspot in time range and add those which within the burned area


# giving an burned_area, and hotspots, returning the hotspots within the area with correct date
def hotspots_in_ba_daily(date_of_fire, burned_area, hotspots_gdf):
    # filter hotspots using temporal
    date_col = pd.to_datetime(hotspots_gdf['date']).dt.date
    temporal_criteria = (date_col == date_of_fire)
    hotspots_gdf_temp = hotspots_gdf[temporal_criteria]
    # filter hotspots again using spatial (more time consuming)
    spatial_criteria = hotspots_gdf_temp.within(burned_area)
    return hotspots_gdf_temp[spatial_criteria]


# create folder for each fire, store a burned area shapefile and seperate shapfiles containing daily hotspots
def daily_data_for_one_fire(ba_row, h_gdf, onefire_path):
    # for each fire, find the hotspot for each day
    start_date = pd.to_datetime(ba_row['sDate'], format='%Y-%m-%d').date()
    end_date = pd.to_datetime(ba_row['eDate'], format='%Y-%m-%d').date()
    fire_timerange = generate_date_list(start_date, end_date)
    # turn off the warnings for saving empty files
    warnings.filterwarnings("ignore", category=UserWarning)
    for date_of_fire in fire_timerange:
        print("  start to do date "+str(date_of_fire)+" of "+str(len(fire_timerange))+" dates")
        daily_h_of_fire = hotspots_in_ba_daily(date_of_fire, ba_row['geometry'], h_gdf)
        # save burned_area and hotspots
        if daily_h_of_fire.empty:
            daily_h_of_fire.to_file(onefire_path + '/' + str(date_of_fire) + '_EMPTY.shp', driver='ESRI Shapefile')
        else:
            daily_h_of_fire.to_file(onefire_path+'/'+str(date_of_fire)+'.shp',driver='ESRI Shapefile')
    warnings.filterwarnings("default", category=UserWarning)
            




burned_area_path = '../../../data/burned_areas/'
hotspot_path = '../../../data/hotspots/'
fire_path = '../../../data/fire/'
years = np.arange(2021, 2022)

pyproj.datadir.set_data_dir('E:/anaconda3/envs/wildfire2/Library/share/proj')
os.environ['PROJ_LIB'] = 'E:/anaconda3/envs/wildfire2/Library/share/proj'

for year in years:
    print("Start to calculate joint of ba and hs in year "+str(year) +'\n')
    if not os.path.exists(fire_path+str(year)):
        os.makedirs(fire_path+str(year))

    ba_gdf = gpd.read_file(burned_area_path+str(year)+'a.shp')
    hotspots_gdf = gpd.read_file(hotspot_path+str(year)+'s.shp')

    yearly_fires = []
    for idx, ba_row in ba_gdf.iterrows():
        print('start to process fire '+str(idx)+'/'+str(len(ba_gdf))+' in year '+str(year))
        fireid = str(year)+'_'+str(idx)
        current_fire_path = fire_path+str(year)+'/'+fireid
        if not os.path.exists(current_fire_path):
            os.makedirs(current_fire_path)
        row_gdf = gpd.GeoDataFrame([ba_row], geometry='geometry', crs=ba_gdf.crs)
        row_gdf.to_file(current_fire_path + '/' + fireid + '.shp',driver='ESRI Shapefile')
        daily_data_for_one_fire(ba_row, hotspots_gdf, current_fire_path)







