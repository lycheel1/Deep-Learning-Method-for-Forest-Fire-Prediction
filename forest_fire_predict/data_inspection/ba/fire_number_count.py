import os
import geopandas as gpd
import pandas as pd
import numpy as np

# Set paths and year range
raw_data_path = "../../../raw_data/burned_areas/"
processed_data_path = "../../../data/burned_areas/"
year_range = np.arange(1994, 2022)

# Initialize a dictionary to hold the counts
fire_counts = {year: {} for year in year_range}


# Function to count fire events per province for each year
def count_fire_events():
    for year in year_range:
        print(f"Counting fire events for {year}")
        yearly_gdf = gpd.read_file(processed_data_path + str(year) + 'a.shp')

        # Count fire events per province
        province_counts = yearly_gdf['AGENCY'].value_counts().to_dict()

        # Combine counts for all 'PC-(something)' into a single 'PC' count
        pc_count = sum(count for agency, count in province_counts.items() if agency.startswith('PC-'))
        province_counts = {k: v for k, v in province_counts.items() if not k.startswith('PC-')}
        province_counts['PC'] = pc_count

        # Store counts in the dictionary
        fire_counts[year] = province_counts
        fire_counts[year]['yearly_total'] = yearly_gdf.shape[0]

    # Convert the dictionary to a DataFrame
    fire_counts_df = pd.DataFrame(fire_counts).fillna(0).astype(int).transpose()

    # Reorder columns to put 'yearly_total' first
    cols = ['yearly_total'] + [col for col in fire_counts_df.columns if col != 'yearly_total']
    fire_counts_df = fire_counts_df[cols]

    # Add the last row for totals over all years
    total_counts = fire_counts_df.sum(axis=0)
    total_counts.name = 'Total'
    fire_counts_df = pd.concat([fire_counts_df, total_counts.to_frame().T])

    # Save the DataFrame to a CSV file
    fire_counts_df.to_csv('./ba_KDE/fire_event_counts_per_province.csv', index_label='Year')


# Run the function to count fire events and save to CSV
count_fire_events()

