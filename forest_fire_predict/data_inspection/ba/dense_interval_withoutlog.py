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

        median_ba = np.median(ba_list)
        average_ba = np.mean(ba_list)
        min_ba = ba_list.min()
        max_ba = ba_list.max()
        intervals.append((year, min_ba, max_ba, median_ba, average_ba))

        x = np.linspace(min_ba, max_ba, 200)
        kde_values = sns.kdeplot(ba_list, gridsize=200).get_lines()[0].get_ydata()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(x, kde_values, label='KDE', color='blue')
        plt.title(f'Burned Area Density for {year}', fontsize=14)
        plt.xlabel('Burned Area (hectares)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./ba_KDE/burned_area_density_{year}.png', dpi=300)
        plt.close()

    save_intervals_to_csv(intervals, 'ba_KDE/yearly_intervals.csv')

# Function to generate total distribution plot with 95% densest interval
def total_distribution_with_95_interval(fraction):
    print(f"Generating total distribution of burned area 1994-2022 with densest {fraction * 100}%")
    gdf = gpd.read_file(raw_data_path + 'nbac_1986_to_2022_20230630.shp')
    ba_list = gdf['ADJ_HA']

    sns.set_style("whitegrid")

    x = np.linspace(ba_list.min(), ba_list.max(), 6000)
    kde_values = sns.kdeplot(ba_list, gridsize=6000).get_lines()[0].get_ydata()
    plt.close()

    best_interval = densest_interval(x, kde_values, fraction)
    lower_bound = best_interval[0]
    upper_bound = best_interval[1]

    plt.figure(figsize=(10, 6))
    plt.plot(x, kde_values, label='KDE', color='blue')
    plt.axvline(lower_bound, color='red', linestyle='--', label=f'{fraction * 100}% Lower Bound')
    plt.axvline(upper_bound, color='red', linestyle='--', label=f'{fraction * 100}% Upper Bound')
    plt.fill_between(x, 0, kde_values, where=(x > lower_bound) & (x < upper_bound), color='blue', alpha=0.3)

    plt.title(f'Burned Area Density 1994-2022 with {fraction * 100}% Densest Interval', fontsize=14)
    plt.xlabel('Burned Area (hectares)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./ba_KDE/burned_area_density_1994_2022_{fraction}.png', dpi=300)
    plt.close()

    # Save the interval to CSV
    total_min = ba_list.min()
    total_max = ba_list.max()
    total_median = np.median(ba_list)
    total_average = np.mean(ba_list)

    # Print values for sanity check
    print(f"Total Min: {total_min}")
    print(f"Total Max: {total_max}")
    print(f"Total Median: {total_median}")
    print(f"Total Average: {total_average}")

    with open('ba_KDE/total_interval_95.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Lower Bound", "Upper Bound"])
        writer.writerow([lower_bound, upper_bound])

    with open('ba_KDE/total_statistics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Min", "Max", "Median", "Average"])
        writer.writerow([total_min, total_max, total_median, total_average])

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
    plt.savefig(f'./ba_KDE/burned_area_density_filtered_1994_2022_95percent.png', dpi=300)
    plt.close()

# Generate plots and save intervals
yearly_distribution()
lower_bound, upper_bound = total_distribution_with_95_interval(0.95)
kde_95_percent_interval(lower_bound, upper_bound)

