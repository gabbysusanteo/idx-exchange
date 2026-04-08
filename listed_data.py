import glob
import pandas as pd

# Getting the list of all listed CSV files in IDX folder.
all_files = glob.glob('CRMLSListing*.csv')
print(all_files)

# Creating a place to store data.
dfs = []
for file in all_files:
    df = pd.read_csv(file)
    # Counts rows.
    print(file, len(df))
    dfs.append(df)

# Takes all monthly housing data and combines it into one dataset.
listed_data = pd.concat(dfs)
print("Total rows:", len(listed_data))

# Show all property types
print("\nAll Property Types:")
print(listed_data['PropertyType'].value_counts())
print("Rows (All property types):", len(listed_data))

# Filter to Residential
residential_data = listed_data[listed_data['PropertyType'] == 'Residential']

# Show ONLY residential
print("\nResidential Only:")
print(residential_data['PropertyType'].value_counts())
print("Rows (Residential):", len(residential_data))

# Save to new CSV
residential_data.to_csv('residential_listed.csv', index=False)