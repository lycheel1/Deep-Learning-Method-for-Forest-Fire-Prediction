import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the year range and agencies
year_range = np.arange(1994, 2022)
agencies = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'QC', 'SK', 'YT',
            'PC-BA', 'PC-EI', 'PC-GL', 'PC-GR', 'PC-JA', 'PC-KG', 'PC-KO', 'PC-LM',
            'PC-NA', 'PC-PA', 'PC-PP', 'PC-PU', 'PC-RE', 'PC-RM', 'PC-TH', 'PC-TN',
            'PC-VU', 'PC-WB', 'PC-WL', 'PC-WP', 'PC-YO']

raw_data_path = "../../../raw_data/burned_areas/"

# Load the data
gdf = gpd.read_file(raw_data_path + 'nbac_1986_to_2022_20230630.shp')

# Filter data for the specified year range
gdf = gdf[gdf['YEAR'].isin(year_range)]

# Initialize a dataframe to hold the counts
density_matrix = pd.DataFrame(0, index=year_range, columns=agencies)

# Count the number of fire events for each year-agency pair
for year in year_range:
    yearly_data = gdf[gdf['YEAR'] == year]
    for agency in agencies:
        count = yearly_data[yearly_data['AGENCY'] == agency].shape[0]
        density_matrix.at[year, agency] = count

density_matrix.fillna(0, inplace=True)

for year in year_range:
    yearly_data = gdf[gdf['YEAR'] == year]
    for agency in agencies:
        count = yearly_data[yearly_data['AGENCY'] == agency].shape[0]
        density_matrix.at[year, agency] = count

# Plot the heatmap
plt.figure(figsize=(20, 12))  # Increased size for better visibility
sns.heatmap(density_matrix, cmap="YlGnBu", annot=False, fmt="d", cbar_kws={'label': 'Number of Fire Events'}, annot_kws={"size": 8})
plt.title('Number of Fire Events per Year by Agency (1994-2021)', fontsize=20)
plt.xlabel('Agency', fontsize=14)
plt.ylabel('Year', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('fire_events_density_matrix.png', dpi=300)
plt.show()
