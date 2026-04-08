import glob
import pandas as pd

# Getting the list of all sold CSV files in IDX folder.
all_files = glob.glob('CRMLSSold*.csv')
print(all_files)

# Creating a place to store data.
dfs = []
for file in all_files:
    df = pd.read_csv(file)
    # Counts rows.
    print(file, len(df))
    dfs.append(df)

# Takes all monthly housing data and combines it into one dataset.
sold_data = pd.concat(dfs)
print("Total rows:", len(sold_data))

# Show all property types
print("\nAll Property Types:")
print(sold_data['PropertyType'].value_counts())
print("Rows (All property types):", len(sold_data))

# Filter to Residential
residential_data = sold_data[sold_data['PropertyType'] == 'Residential']

# Show ONLY residential
print("\nResidential Only:")
print(residential_data['PropertyType'].value_counts())
print("Rows (Residential):", len(residential_data))

# Save to new CSV
residential_data.to_csv('residential_sold.csv', index=False)