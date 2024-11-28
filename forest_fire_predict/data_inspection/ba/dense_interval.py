# Correcting the previous modification to ensure all bounds are transformed to log in both plots.
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv

# Set paths and year range
raw_data_path = "../../../raw_data/burned_areas/"
processed_data_path = "../../../data/burned_areas/"
year_range = np.arange(1994, 2022)

# Function to calculate the densest interval
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
                break

    return best_interval

# Function to save intervals to CSV
def save_intervals_to_csv(intervals, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Year", "Min", "Max", "Median", "Average"])
        for interval in intervals:
            writer.writerow(interval)

        total_min = min(interval[1] for interval in intervals)
        total_max = max(interval[2] for interval in intervals)
        total_median = np.median([interval[3] for interval in intervals])
        total_average = np.mean([interval[4] for interval in intervals])

        writer.writerow(["Total", total_min, total_max, total_median, total_average])

# Function to generate yearly distribution plots
def yearly_distribution():
    intervals = []
    for year in year_range:
        print(f"Generating distribution of burned area for {year}")
        yearly_gdf = gpd.read_file(processed_data_path + str(year) + 'a.shp')
        ba_list = yearly_gdf['ADJ_HA']

        sns.set_style("whitegrid")

        plt.figure(figsize=(10, 6))
        sns.kdeplot(ba_list, label=f'{year} KDE', color='blue')
        plt.title(f'Burned Area Density for {year}', fontsize=14)
        plt.xlabel('Burned Area (hectares)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./ba_KDE/burned_area_density_{year}.png', dpi=300)
        plt.close()

        kde_values = sns.kdeplot(ba_list).get_lines()[0].get_ydata()
        x_values = sns.kdeplot(ba_list).get_lines()[0].get_xdata()

        interval = densest_interval(x_values, kde_values, 0.95)
        intervals.append([year, interval[0], interval[1], np.median(ba_list), np.mean(ba_list)])

    save_intervals_to_csv(intervals, 'ba_KDE/yearly_intervals.csv')

# Function to generate KDE for the overall distribution with 95% densest interval
def total_distribution_with_95_interval(fraction):
    print("Generating total distribution with 95% interval")
    ba_list = []
    for year in year_range:
        yearly_gdf = gpd.read_file(processed_data_path + str(year) + 'a.shp')
        ba_list.extend(yearly_gdf['ADJ_HA'])

    ba_list = np.array(ba_list)
    kde_values = sns.kdeplot(ba_list).get_lines()[0].get_ydata()
    x_values = sns.kdeplot(ba_list).get_lines()[0].get_xdata()

    lower_bound, upper_bound = densest_interval(x_values, kde_values, fraction)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(ba_list, label='Overall KDE', color='blue')
    plt.axvline(lower_bound, color='red', linestyle='--', label='95% Lower Bound')
    plt.axvline(upper_bound, color='red', linestyle='--', label='95% Upper Bound')
    plt.title('Burned Area Density within 95% Densest Interval (1994-2022)', fontsize=14)
    plt.xlabel('Burned Area (hectares)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./ba_KDE/burned_area_density_1994_2022_95percent.png', dpi=300)
    plt.close()

    # Log version
    log_ba_list = np.log(ba_list + 1)
    log_kde_values = sns.kdeplot(log_ba_list).get_lines()[0].get_ydata()
    log_x_values = sns.kdeplot(log_ba_list).get_lines()[0].get_xdata()

    log_lower_bound, log_upper_bound = densest_interval(log_x_values, log_kde_values, fraction)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(log_ba_list, label='Overall Log KDE', color='blue')
    plt.axvline(log_lower_bound, color='red', linestyle='--', label='95% Lower Bound')
    plt.axvline(log_upper_bound, color='red', linestyle='--', label='95% Upper Bound')
    plt.title('Log Burned Area Density within 95% Densest Interval (1994-2022)', fontsize=14)
    plt.xlabel('Log Burned Area (hectares)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./ba_KDE/log_burned_area_density_1994_2022_95percent.png', dpi=300)
    plt.close()

    return lower_bound, upper_bound

# Function to generate KDE for the 95% densest interval
def kde_95_percent_interval(lower_bound, upper_bound):
    print("Generating KDE for 95% densest interval")
    gdf = gpd.read_file(raw_data_path + 'nbac_1986_to_2022_20230630.shp')
    ba_list = gdf['ADJ_HA']

    filtered_ba_list = ba_list[(ba_list >= lower_bound) & (ba_list <= upper_bound)]

    plt.figure(figsize=(10, 6))
    sns.kdeplot(filtered_ba_list, label='Filtered KDE', color='blue')
    plt.axvline(lower_bound, color='red', linestyle='--', label='95% Lower Bound')
    plt.axvline(upper_bound, color='red', linestyle='--', label='95% Upper Bound')
    plt.title('Burned Area Density within 95% Densest Interval (1994-2022)', fontsize=14)
    plt.xlabel('Burned Area (hectares)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./ba_KDE/burned_area_density_filtered_1994_2022_95percent.png', dpi=300)
    plt.close()

    # Log version
    log_filtered_ba_list = np.log(filtered_ba_list + 1)
    log_lower_bound = np.log(lower_bound + 1)
    log_upper_bound = np.log(upper_bound + 1)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(log_filtered_ba_list, label='Filtered Log KDE', color='blue')
    plt.axvline(log_lower_bound, color='red', linestyle='--', label='95% Lower Bound')
    plt.axvline(log_upper_bound, color='red', linestyle='--', label='95% Upper Bound')
    plt.title('Log Burned Area Density within 95% Densest Interval (1994-2022)', fontsize=14)
    plt.xlabel('Log Burned Area (hectares)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./ba_KDE/log_burned_area_density_filtered_1994_2022_95percent.png', dpi=300)
    plt.close()

# Generate plots and save intervals
# yearly_distribution()
lower_bound, upper_bound = total_distribution_with_95_interval(0.95)
kde_95_percent_interval(lower_bound, upper_bound)





