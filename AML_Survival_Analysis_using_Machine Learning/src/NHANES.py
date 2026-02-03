import pandas as pd
import numpy as np
import os

# Define URLs for the 2021-2023 NHANES cycle L
urls = {
    "demographics": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt",
    "bmi": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt",
    "smoking": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQ_L.xpt",
    "activity": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PAQ_L.xpt"
}

# 1. Load the datasets
# pandas can read XPT files directly from the URL
df_demo = pd.read_sas(urls["demographics"])
df_bmi = pd.read_sas(urls["bmi"])
df_smoke = pd.read_sas(urls["smoking"])
df_act = pd.read_sas(urls["activity"])

# 2. Select only the necessary variables to keep the dataframe clean
# SEQN is the unique ID needed for merging
# Standardize column names to uppercase and select variables
for df in [df_demo, df_bmi, df_smoke, df_act]:
    df.columns = df.columns.str.upper()

df_bmi = df_bmi[['SEQN', 'BMXBMI']]
df_smoke = df_smoke[['SEQN', 'SMQ020', 'SMQ040']]
    
# Use a flexible selector for activity in case some are missing
act_requested = ['SEQN', 'PAD790Q', 'PAD810Q', 'PAD680']
df_act = df_act[[c for c in act_requested if c in df_act.columns]]


# 3. Merge the datasets on 'SEQN'
# We use an 'inner' join to keep only participants who appear in all files
merged_df = df_demo.merge(df_bmi, on='SEQN', how='outer') \
                   .merge(df_smoke, on='SEQN', how='outer') \
                   .merge(df_act, on='SEQN', how='outer')

# 4. Basic Data Cleaning (Example)

"""
# NHANES often uses 7 or 9 for "Refused" or "Don't Know" fill with median? no
# merged_df['BMXBMI'] = merged_df['BMXBMI'].fillna(merged_df['BMXBMI'].median())

# Drop rows where BMI is missing 
# merged_df = merged_df.dropna(subset=['BMXBMI'])
"""

# Convert SEQN floats (1.0, 2.0) to integers (1, 2)
merged_df['SEQN'] = merged_df['SEQN'].round(0).astype('float64').astype('Int64')


# Define the categorical columns and their "unknown" codes
cols_to_fix = ['SMQ020', 'SMQ040', 'PAD790Q', 'PAD810Q']
    
# Convert 7 and 9 to NaN so they don't skew the mean/median
for col in cols_to_fix:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].replace([7, 9, 77, 99], np.nan)
        
"""
# Handle Missing Values
# Fill with the Mode (most common answer 'repeated answer')
for col in cols_to_fix:
    mode_val = merged_df[col].mode()[0]
    merged_df[col] = merged_df[col].fillna(mode_val)
"""

# For Activity (PAD680 - minutes of sedentary behavior), fill with median
merged_df['PAD680'] = merged_df['PAD680'].fillna(merged_df['PAD680'].median()).round(0)

# Example: Filter out physiologically impossible BMI values
# (e.g., keep only values between 10 and 80)
merged_df = merged_df[(merged_df['BMXBMI'] > 10) & (merged_df['BMXBMI'] < 80)]

# Check for and drop exact duplicates of rows 
merged_df = merged_df.drop_duplicates()

## Sort, Fill, and Drop
# Sort the data so rows with fewer NaNs appear first to pick the most complete data
merged_df['nan_count'] = merged_df.isnull().sum(axis=1)
merged_df = merged_df.sort_values(by=['SEQN', 'nan_count'])

# Group by SEQN and use 'first()'
# .first() automatically picks the first NON-NULL value it finds for each column
merged_df = merged_df.groupby('SEQN').first().reset_index()

# Clean up: Remove the helper column
merged_df = merged_df.drop(columns=['nan_count'])

# verify that every SEQN is now unique
assert merged_df['SEQN'].is_unique

# Ensure SEQN is truly unique (no two rows for the same person)
#merged_df = merged_df.drop_duplicates(subset=['SEQN'])

# Convert categorical floats (1.0, 2.0) to integers (1, 2)
for col in cols_to_fix:
    if col in merged_df.columns:
        # 1. Round the values to 0 decimal places
        # 2. Use .astype('float64') first to ensure consistency
        # 3. Finally, convert to 'Int64'
        # We use 'Int64' (capital I) because it supports NaN values
        merged_df[col] = merged_df[col].round(0).astype('float64').astype('Int64')

# Final Printout
print("-" * 30)
print(f"Final dataset shape: {merged_df.shape}")
print("-" * 30)
print(merged_df.head(40))

print("-" * 100)
missing_bmi = merged_df[merged_df['BMXBMI'].isna()]
print(f"Number of rows BMI values missing: {len(missing_bmi)}")
print("-" * 100)
print(" " )

missing_data = merged_df[merged_df['SMQ020'].isna() & merged_df['PAD680'].isna()]

print(f"Number of rows with both values missing: {len(missing_data)}")
print("-" * 100)
print(missing_data[['SEQN','SMQ020','PAD680']])


# Save the cleaned data to CSV file

output_dir = '../data/processed/'
output_filename = "NHANES_Cleaned.csv"
full_output_path = os.path.join(output_dir, output_filename)

# Ensures the output direectory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Export the DataFrame

merged_df.to_csv(full_output_path, index=False, na_rep='NaN')

print(f"✅ Success! Your data has been exported to: {output_filename}")


