import os
import geopandas as gpd
import pandas as pd
import numpy as np

# Set paths and year range
hotspot_data_path = "../../../data/hotspots/"
year_range = np.arange(1994, 2022)

# Initialize a dictionary to hold the counts
hotspot_counts = {year: {} for year in year_range}


# Function to count hotspot events per agency for each year
def count_hotspot_events():
    for year in year_range:
        print(f"Counting hotspot events for {year}")
        file_path = os.path.join(hotspot_data_path, f"{year}s.shp")
        yearly_gdf = gpd.read_file(file_path)

        # Count hotspot events per agency
        agency_counts = yearly_gdf['agency'].value_counts().to_dict()

        # Store counts in the dictionary
        hotspot_counts[year] = agency_counts
        hotspot_counts[year]['yearly_total'] = yearly_gdf.shape[0]

    # Convert the dictionary to a DataFrame
    hotspot_counts_df = pd.DataFrame(hotspot_counts).fillna(0).astype(int).transpose()

    # Add the last row for totals over all years
    total_counts = hotspot_counts_df.sum(axis=0)
    total_counts.name = 'Total'
    hotspot_counts_df = pd.concat([hotspot_counts_df, total_counts.to_frame().T])

    # Reorder columns to put 'yearly_total' first
    cols = ['yearly_total'] + [col for col in hotspot_counts_df.columns if col != 'yearly_total']
    hotspot_counts_df = hotspot_counts_df[cols]

    # Save the DataFrame to a CSV file
    hotspot_counts_df.to_csv('./hotspot_counts_per_agency.csv', index_label='Year')


# Run the function to count hotspot events and save to CSV
count_hotspot_events()
