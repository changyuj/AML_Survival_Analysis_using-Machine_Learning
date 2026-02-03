import pandas as pd
import numpy as np
import os


interested_columns = [
    'Patient ID', 
    'Sex', 
    'Age at Diagnosis', 
    'Cancer Type', 
    'Overall Survival Status', 
    'Overall Survival (Months)'
    ]
    
# 1. Load the patient data from TSV file 
#df = pd.read_csv('aml_ohsu_2022_clinical_data.tsv', sep='\t', usecols=interested_columns)
df = pd.read_table('../data/raw/aml_ohsu_2022_clinical_data.tsv', usecols=interested_columns)



# 2. Clean column names
# -1. replace spaces with underscores
# -2. Remove parentheses entirely
# -3. Convert to lowercase

df.columns = [c.replace(' ', '_').replace('(', '').replace(')','').lower() for c in df.columns]

# 3. Standardize data

# Check for and drop exact duplicates of rows 
df = df.drop_duplicates()

# Count missing values for each row so we can find the fullest ones
df['null_count'] = df.isnull().sum(axis=1)

# sort by patient_id, then by null_count (ascending)
df = df.sort_values(by=['patient_id', 'null_count'])

# Use groupby and 'first()' to fill the holes
df_clean = df.groupby('patient_id').first().reset_index()

# Remove the helper column
df_clean = df_clean.drop(columns=['null_count'])

# Round to 0 decimal places
df_clean['overall_survival_months'] = df_clean['overall_survival_months'].round(2)

#  Use .astype('float64') first to ensure consistency and convert to 'Int64'
df_clean['age_at_diagnosis'] = df_clean['age_at_diagnosis'].astype('float64').astype('Int64')

# Slicing [3:] means: "Start at the 4th character and take everything until the end"
df_clean['overall_survival_status'] = df_clean['overall_survival_status'].str[2:]

# Optional: Strip any accidental leftover whitespace just in case
df_clean['overall_survival_status'] = df_clean['overall_survival_status'].str.strip()

print("-" * 50)
print("cleaned overall survivial status")
print(df_clean['overall_survival_status'].unique())
print("-" * 50)

# Standardize the column (just in case there are weird spaces or capitalization)
df_clean['cancer_type'] = df_clean['cancer_type'].str.strip()

# Filter to keep only the target type
df_clean = df_clean[df_clean['cancer_type'] == "Acute Myeloid Leukemia"]

# Verify the result
print("-" * 50)
print(f"Remaining rows: {len(df_clean)}")
print(df_clean['cancer_type'].unique())
print("-" * 50)

print("-" * 50)
print(f"Original rows: {len(df)}")
print(f"Consolidated rows: {len(df_clean)}")
print("-" * 50)

# Display the frist 2 rows and data info (checking successful read)
print(df_clean.head(10))
print(df_clean.info())

# Save the cleaned data to CSV file

output_dir = '../data/processed/'
output_filename = "patient_data_cleaned.csv"
full_output_path = os.path.join(output_dir, output_filename)

# Ensures the output direectory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_clean.to_csv(full_output_path, index=False, na_rep='NaN')

print(f"✅ Success! Your data has been exported to: {output_filename}")