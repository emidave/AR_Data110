import pandas as pd
import numpy as np

# Importing dataframes with all countries and the relevant variables
resistance = pd.read_csv('resistance.csv')
antibiotic_use_livestock = pd.read_csv('antibiotic_use_livestock.csv')
antibiotics_sold_without_prescription = pd.read_csv('antibiotics_sold_without_prescription.csv')
egg_consumption = pd.read_csv('egg_consumption.csv')
expected_schooling_years = pd.read_csv('expected_schooling_years.csv')
ibs_rates = pd.read_csv('ibs_rates.csv')
meat_consumption = pd.read_csv('meat_consumption.csv')
milk_consumption = pd.read_csv('milk_consumption.csv')

# Mapping guide
all_map = pd.read_csv('country_region_dictionary.csv')
all_map = pd.DataFrame(all_map)

# Define the mapping of countries to regions (dictionary)
def map_updated_region(region):
    if pd.isna(region):  # Check if the value is NaN
        return region  # Return NaN or handle it as needed
    if region == 'West Asia':
        return 'Middle East and North Africa'
    elif region in ['Australia and New Zealand', 'Eastern Asia']:
        return 'East Asia and Pacific'
    elif 'Central Asia' in region or 'Europe' in region:
        return 'Europe and Central Asia'
    elif region in ['Southern Asia', 'South-eastern Asia']:
        return 'South Asia'
    elif region in ['Micronesia', 'Polynesia']:
        return 'East Asia and Pacific'
    elif region in ['Western Asia', 'Northern Africa']:
        return 'Middle East and North Africa'
    else:
        return region  # Default to the original sub-region if no match

# Apply the mapping function to create the 'updated_region' column
#all_map['updated_region'] = all_map['region'].apply(map_updated_region)

# Creating a dictionary from the 'name' and 'updated_region' columns
country_to_updated_region = dict(zip(all_map['name'], all_map['region']))

# Function to add a region column to a dataframe based on dictionary (handles substrings and case-insensitivity)
def add_region_column(df):
    def map_region(country):
        for key, region in country_to_updated_region.items():
            if key.lower() in country.lower():
                return region
        return "Empty"  # Return None if no match is found

    df['Region'] = df['Country'].apply(map_region)
    return df

def update_country_column(df):
    def update_country_name(country):
        for key in country_to_updated_region.keys():
            if key.lower() in country.lower() and key.lower() != country.lower():
                return key  # Replace with the key substring
        return country  # Return the original value if no match is found

    df['Country'] = df['Country'].apply(update_country_name)
    return df

# Revising country column for each df
resistance = update_country_column(resistance)
antibiotic_use_livestock = update_country_column(antibiotic_use_livestock)
antibiotics_sold_without_prescription = update_country_column(antibiotics_sold_without_prescription)
egg_consumption = update_country_column(egg_consumption)
expected_schooling_years = update_country_column(expected_schooling_years)
ibs_rates = update_country_column(ibs_rates)
meat_consumption = update_country_column(meat_consumption)
milk_consumption = update_country_column(milk_consumption)

# Now adding the region columns to each of our dataframes
resistance = add_region_column(resistance)
antibiotic_use_livestock = add_region_column(antibiotic_use_livestock)
antibiotics_sold_without_prescription = add_region_column(antibiotics_sold_without_prescription)
egg_consumption = add_region_column(egg_consumption)
expected_schooling_years = add_region_column(expected_schooling_years)
ibs_rates = add_region_column(ibs_rates)
meat_consumption = add_region_column(meat_consumption)
milk_consumption = add_region_column(milk_consumption)

# Removing the inapplicable rows or countries without an associated region
def remove_empty_regions(df):
    return df[df['Region'] != "Empty"]

resistance = remove_empty_regions(resistance)
antibiotic_use_livestock = remove_empty_regions(antibiotic_use_livestock)
antibiotics_sold_without_prescription = remove_empty_regions(antibiotics_sold_without_prescription)
egg_consumption = remove_empty_regions(egg_consumption)
expected_schooling_years = remove_empty_regions(expected_schooling_years)
ibs_rates = remove_empty_regions(ibs_rates)
meat_consumption = remove_empty_regions(meat_consumption)
milk_consumption = remove_empty_regions(milk_consumption)

# Initialize merged_df from the all_map dictionary
merged_df = pd.DataFrame(list(country_to_updated_region.items()), columns=['Country', 'Region'])

def expand_merged_df(input_df, column_name):
    global merged_df  # Use the global merged_df
    input_df['Country'] = input_df['Country'].str.lower()  # Normalize to lowercase for case-insensitive matching
    merged_df['Country'] = merged_df['Country'].str.lower()  # Normalize merged_df as well

    # Create a new column in merged_df with default value "Empty"
    merged_df[column_name] = "Empty"

    # Iterate through the rows of the input dataframe
    for _, row in input_df.iterrows():
        input_country = row['Country']
        input_value = row[input_df.columns[1]]  # Get the value from the non-Country column

        # Check for matches in merged_df
        for i, merged_row in merged_df.iterrows():
            if merged_row['Country'] == input_country or merged_row['Country'] in input_country:
                merged_df.at[i, column_name] = input_value  # Update the new column with the matched value

    # Restore original case for merged_df
    merged_df['Country'] = merged_df['Country'].str.title()

# Example usage with ibs_rates
expand_merged_df(antibiotic_use_livestock, 'antimicrobial_mg_per_population')  # Add the antibiotic_use_livestock column to merged_df
expand_merged_df(egg_consumption, 'kilograms_eggs_per_year_per_capita')  # Add the egg_consumption column to merged_df
expand_merged_df(expected_schooling_years, 'expected_schooling_years')  # Add the expected_schooling_years column to merged_df
expand_merged_df(ibs_rates, 'ibs_rate_2017')  # Add the ibs_rate_2017 column to merged_df
expand_merged_df(meat_consumption, 'kilograms_meat_per_year_per_capita')  # Add the meat_consumption column to merged_df
expand_merged_df(milk_consumption, 'kilograms_milk_per_year_per_capita')  # Add the milk_consumption column to merged_df
expand_merged_df(resistance, 'PercentResistant')  # Add the resistance column to merged_df

print(merged_df.head())
merged_df.to_csv('merged.csv', index=False)

# Calculate summary statistics for each column variable grouped by Region
def calculate_summary_statistics(df):
    # Replace "Empty" with NaN to ensure it's ignored in calculations
    df = df.replace("Empty", np.nan)

    # Exclude the 'Country' column from the statistics
    numeric_columns = df.drop(columns=['Country']).select_dtypes(include=[np.number, 'float', 'int']).columns

    # Group by 'Region' and calculate summary statistics
    summary_stats = df.groupby('Region')[numeric_columns].agg(['mean', 'median', 'min', 'max', 'std', 'count'])

    return summary_stats

# Example usage
summary_statistics = calculate_summary_statistics(merged_df)

# Save the summary statistics to a CSV file
summary_statistics.to_csv('summary_statistics_by_region.csv')

print(summary_statistics)
