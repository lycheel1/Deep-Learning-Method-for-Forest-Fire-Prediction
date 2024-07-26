import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import seaborn as sns
from scipy.stats import gaussian_kde

raw_data_path = "../raw_data/burned_areas/"
processed_data_path = "../data/burned_areas/"
year_range = np.arange(1994,2023)

pyproj.datadir.set_data_dir('E:/anaconda3/envs/wildfire2/Library/share/proj')
os.environ['PROJ_LIB'] = 'E:/anaconda3/envs/wildfire2/Library/share/proj'

def densest_interval(x, kde_values, fraction):
    total_area = np.trapz(kde_values, x)
    target_area = total_area * fraction

    max_density = 0
    best_interval = (x[0], x[-1])

    for i in range(len(x)):
        for j in range(i, len(x)):
            area = np.trapz(kde_values[i:j + 1], x[i:j + 1])
            if area >= target_area:
                interval_length = x[j] - x[i]
                density = area / interval_length
                if density > max_density:
                    max_density = density
                    best_interval = (x[i], x[j])
                break  # Once we've found an interval that covers the target area, move to the next starting point

    return best_interval

def yearly_distribution(fraction):
    ### yearly distribution
    # read yearly burned area
    for year in year_range:
        print("generating distribution of burned area "+str(year)+" with densest " + str(fraction*100) +"%")
        yearly_gdf = gpd.read_file(processed_data_path + str(year) + 'a.shp')
        ba_list = yearly_gdf['ADJ_HA']
        sns.set_style("whitegrid")

        # get 80% boundary
        x = np.linspace(ba_list.min(), ba_list.max(), 200)
        # kde = gaussian_kde(ba_list)
        # kde_values = kde.evaluate(x)
        kde_values = sns.kdeplot(ba_list, gridsize=200).get_lines()[0].get_ydata()
        plt.close()  # Close the initial plot

        # Find x-values for 10% and 90% quantiles (for 80% interval)
        best_interval = densest_interval(x, kde_values, fraction)
        lower_bound = best_interval[0]
        upper_bound = best_interval[-1]

        # Plotting the distribution chart (histogram)
        plt.figure(figsize=(15, 6))
        plt.plot(x, kde_values, label='KDE', color='blue')
        plt.axvline(lower_bound, color='red', linestyle='--', label='10% Quantile')
        plt.axvline(upper_bound, color='red', linestyle='--', label='90% Quantile')

        plt.fill_between(x, 0, kde_values, where=(x > lower_bound) & (x < upper_bound), color='blue', alpha=0.5,
                         label='80% Interval')
        # Annotate the plot with the x-values of the boundaries
        plt.text(lower_bound, 0, f'{int(lower_bound)}', color='red', ha='center', va='bottom', fontsize=8)
        plt.text(upper_bound, 0, f'{int(upper_bound)}', color='red', ha='center', va='bottom', fontsize=8)

        plt.title('burned_area_density ' + str(year) + ' KDE plot with '+ str(fraction*100) +'% densest interval (' + str(
            len(ba_list)) + ' in total)')
        plt.xlabel('Value')
        plt.ylabel('Density')

        # Save the plot as an image
        plt.savefig('ba_' + str(year) + '_' + str(fraction) +'.png', dpi=300)
        plt.show()

def total_distribution(fraction):
    print("generating distribution of burned area 1994-2022" + " with densest " + str(fraction * 100) + "%")
    ### total distribution
    gdf = gpd.read_file(raw_data_path + 'nbac_1986_to_2022_20230630.shp')
    ba_list = gdf['ADJ_HA']
    sns.set_style("whitegrid")

    # get 80% boundary
    x = np.linspace(ba_list.min(), ba_list.max(), 6000)
    # kde = gaussian_kde(ba_list)
    # kde_values = kde.evaluate(x)
    kde_values = sns.kdeplot(ba_list, gridsize=6000).get_lines()[0].get_ydata()
    plt.close()  # Close the initial plot

    # Find x-values for 10% and 90% quantiles (for 80% interval)
    best_interval = densest_interval(x, kde_values, fraction)
    lower_bound = best_interval[0]
    upper_bound = best_interval[-1]

    # Plotting the distribution chart (histogram)
    plt.figure(figsize=(30, 6))
    plt.plot(x, kde_values, label='KDE', color='blue')
    plt.axvline(lower_bound, color='red', linestyle='--', label='10% Quantile')
    plt.axvline(upper_bound, color='red', linestyle='--', label='90% Quantile')

    plt.fill_between(x, 0, kde_values, where=(x > lower_bound) & (x < upper_bound), color='blue', alpha=0.5,
                     label='80% Interval')
    # Annotate the plot with the x-values of the boundaries
    plt.text(lower_bound, 0, f'{int(lower_bound)}', color='red', ha='center', va='bottom', fontsize=8)
    plt.text(upper_bound, 0, f'{int(upper_bound)}', color='red', ha='center', va='bottom', fontsize=8)

    plt.title('burned_area_density from 1994 to 2022' + ' KDE plot with '+ str(fraction*100) +'% densest interval (' + str(
        len(ba_list)) + ' in total)')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Save the plot as an image
    plt.savefig('ba_1994_2022' + '_' + str(fraction) +'.png', dpi=300)


# yearly_distribution(0.8)
# total_distribution(0.8)
#
# yearly_distribution(0.9)
# total_distribution(0.9)

yearly_distribution(0.95)
total_distribution(0.95)



