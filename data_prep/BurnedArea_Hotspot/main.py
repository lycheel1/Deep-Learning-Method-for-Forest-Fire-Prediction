from DataProcessor import DataProcessor
import numpy as np
import os
import pyproj

""" Seperate ba file to yearly ba files"""

pyproj.datadir.set_data_dir('E:/anaconda3/envs/wildfire2/Library/share/proj')
os.environ['PROJ_LIB'] = 'E:/anaconda3/envs/wildfire2/Library/share/proj'

data_processor = DataProcessor()
raw_data_path = "../../raw_data"
processed_data_path = "../../data"

year_range = np.arange(1994,2023)

# Hotspots processing
if True:
    print("start to process hotspots")
    for i in year_range:
        data_processor.date_to_hotspots(raw_data_path+"/hotspots/"+str(i)+"_hotspots.shp", processed_data_path+"/hotspots/"+str(i)+"s.shp")
        print("Task in progress: Hotspots in "+str(i))
    print("TASK DONE: Hotspots date insertion")

# Burned area processing
# There is a warning about requesting Datetime from Date type (sDate eDate) in function .to_file()
# It could be by the limitation of .shp file that it must read Datetime type?
if True:
    print("start to process burned areas")
    data_processor.split_year_areas(raw_data_path + "/burned_areas/" + "nbac_1986_to_2022_20230630.shp", year_range,
                                    raw_data_path + "/burned_areas", "a_raw")
    for i in year_range:
        data_processor.date_to_burned_areas(raw_data_path + "/burned_areas/" + str(i)+"a_raw.shp",
                                        processed_data_path + "/burned_areas/"+ str(i)+"a.shp")

    print("Finished burned areas in 1986-2023")

























    ### reverse the new version
    # data_processor.date_to_burned_areas(raw_data_path + "/burned_areas/" + "nbac_1986_to_2022_20230630.shp",
    #                                     raw_data_path + "/burned_areas/" + "nbac_1986_to_2022_processed.shp")
    # data_processor.split_year_areas(raw_data_path + "/burned_areas/" + "nbac_1986_to_2022_processed.shp", year_range,
    #                                  processed_data_path + "/burned_areas", "a")


    ### OLD version ###

    # data_processor.date_to_burned_areas(raw_data_path+"/burned_areas/"+"nbac_1986_to_2020_20210810.shp", processed_data_path+"/burned_areas/1986-2020.shp")
    # data_processor.split_year_areas(processed_data_path+"/burned_areas/1986-2020.shp", np.arange(1986,2021), processed_data_path+"/burned_areas", "a")
    # print("Task in progress burned areas in 1986-2020")

    # data_processor.date_to_burned_areas(raw_data_path+"/burned_areas/"+"nbac_2021_r9_20220624.shp", processed_data_path+"/burned_areas/2021-2022.shp")
    # data_processor.split_year_areas(processed_data_path+"/burned_areas/2021-2022.shp", np.arange(2021,2023), processed_data_path+"/burned_areas", "a")
    # print("Task in progress burned areas in 2021-2022")
    # print("TASK DONE: Burned area date insertion")

