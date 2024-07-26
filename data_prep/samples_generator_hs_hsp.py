import os

import numpy as np
import pandas as pd
import geopandas as gpd

from util.date_list import generate_date_list
from datetime import timedelta

from shutil import copyfile
from FWI.fwi_NaN_check import contains_NaN

def generate_dataset_hs_hsp():
    fire_instances_path = '../data/fire instances 64 200'
    ds_samples_path = '../data/ds samples 64 200'
    years = np.arange(1994, 2022)

    # create a df for storing all the samples' meta
    meta_df = pd.DataFrame(columns=['SAMPLE_INDEX','fireID','NFIREID','YEAR','AGENCY','FIRECAUS','ADJ_HA','bDays','sDate','eDate','is_mp'])
    # sample count
    sample_index = 0

    for year in years:
        print(f"\nstart to generate samples in {year}")

        yearly_instances_path = f'{fire_instances_path}/{year}'

        for fire_instance_folder in os.listdir(yearly_instances_path):
            print(f"\nfire {fire_instance_folder}")
            fire_instance_path = f'{yearly_instances_path}/{fire_instance_folder}/'
            ba_gdf = gpd.read_file(f'{fire_instance_path}/{fire_instance_folder} ba.shp')
            ba_row = ba_gdf.iloc[0]

            start_date = pd.to_datetime(ba_row['sDate'], format='%Y-%m-%d').date()
            end_date = pd.to_datetime(ba_row['eDate'], format='%Y-%m-%d').date()
            fire_timerange = generate_date_list(start_date, end_date - timedelta(days=1))

            for date in fire_timerange:
                # find the hotspots for input and output if they exist
                todays_hs = f'{fire_instance_path}/{fire_instance_folder} {str(date)} hs.npy'
                tmrs_hs = f'{fire_instance_path}/{fire_instance_folder} {str(date+timedelta(days=1))} hs.npy'
                todays_fwi = f'{fire_instance_path}/{fire_instance_folder} {str(date)} fwi.npy'

                # fwi is the only feature that could still contains nan afte masking and can't be fixed
                if os.path.isfile(todays_hs) and os.path.isfile(tmrs_hs) and (not contains_NaN(todays_fwi)):
                    print(f"found {sample_index}th sample: {str(date)}")

                    current_sample_path = f'{ds_samples_path}/{sample_index}'
                    if not os.path.exists(current_sample_path):
                        os.makedirs(current_sample_path)

                    #### start copy files ###

                    # hotspot day i
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} {str(date)} hs.npy',
                             f'{current_sample_path}/{sample_index} hs.npy')
                    # hotspot day i, cumulative
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} {str(date)} chs.npy',
                             f'{current_sample_path}/{sample_index} chs.npy')
                    # hotspot day i, polygon
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} {str(date)} hs_poly.npy',
                             f'{current_sample_path}/{sample_index} hsp.npy')
                    # hotspot day i, cumulative, polygon
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} {str(date)} chs_poly.npy',
                             f'{current_sample_path}/{sample_index} chsp.npy')
                    # hotspot day i+1
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} {str(date+timedelta(days=1))} hs.npy',
                             f'{current_sample_path}/{sample_index} Y.npy')
                    # hotspot day i+1, cumulative
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} {str(date+timedelta(days=1))} chs.npy',
                             f'{current_sample_path}/{sample_index} Yc.npy')
                    # hotspot day i+1, polygon
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} {str(date + timedelta(days=1))} hs_poly.npy',
                             f'{current_sample_path}/{sample_index} Yp.npy')
                    # hotspot day i+1, cumulative, polygon
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} {str(date + timedelta(days=1))} chs_poly.npy',
                             f'{current_sample_path}/{sample_index} Ycp.npy')

                    # elevation
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} elev.npy',
                             f'{current_sample_path}/{sample_index} elev.npy')
                    # fuel
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} fuel.npy',
                             f'{current_sample_path}/{sample_index} fuel.npy')
                    # fire weather index day i
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} {str(date)} fwi.npy',
                             f'{current_sample_path}/{sample_index} fwi.npy')
                    # agency (provinces or park, etc)
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} agency.npy',
                             f'{current_sample_path}/{sample_index} agency.npy')
                    # fire causes
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} fc.npy',
                             f'{current_sample_path}/{sample_index} fc.npy')
                    # multipolygon or not
                    copyfile(f'{fire_instance_path}/{fire_instance_folder} mp.npy',
                             f'{current_sample_path}/{sample_index} mp.npy')

                    # create month 1D feature
                    one_hot_encoded_month = np.zeros(12)
                    one_hot_encoded_month[date.month - 1] = 1
                    np.save(f'{current_sample_path}/{sample_index} month.npy', one_hot_encoded_month)

                    # update dataframe 'SAMPLE_INDEX','fireID','NFIREID','YEAR','AGENCY','FIRECAUS','ADJ_HA','bDays','sDate','eDate','is_mp'
                    new_row_data = ([sample_index, fire_instance_folder] + ba_row[['NFIREID', 'YEAR', 'AGENCY', 'FIRECAUS', 'ADJ_HA', 'bDays','sDate','eDate','is_mp']].tolist())
                    new_row_df = pd.DataFrame([new_row_data], columns=meta_df.columns)
                    meta_df = pd.concat([meta_df, new_row_df], ignore_index=True)

                    sample_index += 1

    meta_df.to_csv(f'{ds_samples_path}/samples_meta.csv', index=False)
    print(f"dataframe saved, {sample_index} samples in total (starting from 0)")


generate_dataset_hs_hsp()





