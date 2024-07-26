# import shapefile as SF
import geopandas as gpd
# import geodatasets as gds
import pandas as pd
# import matplotlib.pyplot as plt
import osgeo
import fiona
from shapely.geometry import MultiPolygon, Polygon


class DataProcessor:
    def __init__(self):
        self.hotspots = None
        self.burned_area = None

    def convert_to_multipolygon(self, geom):
        if isinstance(geom, Polygon):
            return MultiPolygon([geom])
        else:
            return geom

    # Take raw M3 hotspots data (.shp)
    def date_to_burned_areas(self, file_path, output_path):
        df = gpd.read_file(file_path)

        # transfrom all column name into uppercase and filter them
        df.columns = df.columns.map(lambda x: x.upper())
        df_filtered = df.loc[:, ["NFIREID", "YEAR", "SDATE", "EDATE", "AFSDATE", "AFEDATE", "AGENCY", "FIRECAUS", "ADJ_HA", "GEOMETRY"]]
        df_filtered = df_filtered.rename(columns={'GEOMETRY': 'geometry'})
        # transform date in object types to Date type
        for col in ["SDATE", "EDATE", "AFSDATE", "AFEDATE"]:
            df_filtered[col] = pd.to_datetime(df_filtered[col], format='%Y-%m-%d')

        # the dates in data file have random missing part
        df_filtered.loc[:, "sDate"] = df_filtered.loc[:, ["SDATE", "EDATE", "AFSDATE", "AFEDATE"]].min(axis=1, skipna=True)
        df_filtered.loc[:, "eDate"] = df_filtered.loc[:, ["SDATE", "EDATE", "AFSDATE", "AFEDATE"]].max(axis=1, skipna=True)

        # if all four dats missing then some row still contain NAN
        df_filtered = df_filtered.drop(columns=["SDATE", "EDATE", "AFSDATE", "AFEDATE"])

        # drop dates first because the condition is all four dates are NaN not any
        df_filtered = df_filtered.dropna(axis=0, how='any')

        #
        df_filtered['is_mp'] = df_filtered['geometry'].apply(lambda x: 1 if x.geom_type == 'MultiPolygon' else 0)

        try:
            df_filtered.loc[:, "bDays"] = (df_filtered.loc[:, "eDate"] - df_filtered.loc[:, "sDate"]).dt.days.astype(
                int)
        except:
            print("burned area still contain Nan for the file: " + file_path)

        # only keep fires that length bigger than 1 day
        df_filtered = df_filtered[df_filtered['bDays'] != 0]
        df_filtered = df_filtered.sort_values(by="sDate", axis=0)

        df_filtered["sDate"] = df_filtered["sDate"].dt.strftime('%Y-%m-%d')
        df_filtered["eDate"] = df_filtered["eDate"].dt.strftime('%Y-%m-%d')
        # print(df_filtered.dtypes)

        # check if any invalid lines
        invalid = df_filtered[~df_filtered.is_valid]
        print(invalid)


        # df_filtered["geometry"] = df_filtered['geometry'].apply(self.convert_to_multipolygon)
        # df.set_geometry('geometry', inplace=True)
        df_filtered.to_file(output_path,driver='ESRI Shapefile')

    # Take raw burned areas polygon data (.shp)
    def date_to_hotspots(self, file_path, output_path):
        df = gpd.read_file(file_path)
        df.columns = df.columns.map(lambda x: x.lower())
        df_filtered = df.loc[:, ["rep_date",'agency','fwi','fuel', "geometry"]]

        df_filtered.loc[:, "datetime"] = pd.to_datetime(df_filtered.loc[:, "rep_date"])
        # sort by date and time
        df_filtered = df_filtered.sort_values(by="datetime", axis=0)

        df_filtered.loc[:, "date"] = df_filtered.loc[:, "datetime"].dt.date.astype(str)
        df_filtered.loc[:, "time"] = df_filtered.loc[:, "datetime"].dt.time.astype(str)

        df_filtered = df_filtered.drop(columns=["rep_date", "datetime"])
        df_filtered = df_filtered.dropna(axis=0, subset=["geometry", "date"])

        df_filtered.to_file(output_path)

    # the data to be split should be sorted
    def split_year_areas(self, file_path, range, output_folder, output_name):
        df = gpd.read_file(file_path)
        try:
            for i in range:
                df_i = df.loc[df.loc[:, "YEAR"] == i]
                df_i.to_file(output_folder + '/' + str(i) + output_name + ".shp", driver='ESRI Shapefile')
                print("Burned area in year "+str(i)+" split")
        except:
            print("Failed to read attribute YEAR from burned areas, please make sure the file is correct")

    def split_year_hotspots(self, file_path, range, output_folder, output_name):
        df = gpd.read_file(file_path)
        try:
            for i in range:
                df_i = df.loc[df.loc[:, "year"] == i]
                df_i.to_file(output_folder + '/' + str(i) + output_name + ".shp")
        except:
            print("Failed to read attribute year from hotspots, please make sure the file is correct")

    def grid_preprocess(self, polygon_file, label_file, output_path):
        poly = gpd.read_file(polygon_file)
        label = gpd.read_file(label_file)
        poly.sort_values(by="OID", inplace=True)
        label.sort_values(by="OID", inplace=True)
        df = label.loc[:, "OID"]
        df.loc[:, "mid_point"] = label.loc[:, "geometry"]
        df.loc[:, "area"] = label.loc[:, "geometry"]

        df.tofile(output_path)

    # [OID mid_point area] & [NFIREID YEAR sDate eDate AGENCY GEOMETRY]
    def rasterize_polygon(self, polygon_file, raster_file):
        poly = gpd.read_file(polygon_file)
        raster = gpd.read_file(raster_file)
        rastered_poly = gpd.sjoin(raster, poly, op='intersect', how='left')

    # [OID mid_point area] & [Date Time year geometry]
    def rasterize_point(self, point_file, raster_file):
        point = gpd.read_file(point_file)
        raster = gpd.read_file(raster_file)
        rastered_point = gpd.sjoin(raster, point, op='intersect', how='left')
