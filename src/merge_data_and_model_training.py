import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

""" 1. Import and Load Clinical Data""" 

# load cleaned clinical patient data file
aml_df = pd.read_csv("../data/processed/patient_data_cleaned.csv")

# Filter for the columns we need
cols = ['patient_id', 'age_at_diagnosis', 'sex', 'overall_survival_status', 'overall_survival_months']
aml_df = aml_df[cols].dropna()

""" remove "is_reitred", this is duplicate to age and cause model confustion

# 1. Feature Engineering: Retirement Age
# We define retirement as age 65+
aml_df['is_retired'] = np.where(aml_df['age_at_diagnosis'] >= 65, 1, 0)

"""

""" 2. Load Lifestyle Data (NHANES Proxy)"""
# Load NHANES lifestyle data
# NHANES variables: SEQN (ID), SMQ020 (Smoking), BMXBMI (BMI), PAD680 (Sedentary time)
nhanes_df = pd.read_csv("../data/processed/NHANES_Cleaned.csv")

# Create a "Lifestyle Risk Index" (0 to 3, where higher is worse)
# Example: 1 point if BMI > 30, 1 if smoker, 1 if sedentary > 8 hours
nhanes_df['lifestyle_score'] = ((nhanes_df['BMXBMI'] > 30).astype(int) + (nhanes_df['SMQ020'] == 1).astype(int))


""" 3. The "Statistical Merge" """

# Define the age bins
bins = [0, 30, 45, 60, 75, 120]
labels = ['Young', 'Adult', 'Mid-Age', 'Senior', 'Elderly']

# Create age groups in NHANES
nhanes_df['age_group'] = pd.cut(nhanes_df['RIDAGEYR'], bins=bins, labels=labels)

# Group by Age Group and Sex to get a 'Lifestyle Probability'
# This breaks the direct 1:1 correlation with exact age
lifestyle_map = nhanes_df.groupby(['age_group', 'RIAGENDR'], observed=False)['lifestyle_score'].mean().reset_index()

# Rename columns IMMEDIATELY after reset_index() 
# This changes 'RIAGENDR' to 'sex' so the next lines work
lifestyle_map.columns = ['age_group', 'sex', 'predicted_lifestyle_risk']

# Merge the datasets
# 1. First, make sure the column is an integer so the mapping works
lifestyle_map['sex'] = lifestyle_map['sex'].astype(int)

# 2. Map the NHANES numbers to the AML strings
gender_map = {1: 'Male', 2: 'Female'}
lifestyle_map['sex'] = lifestyle_map['sex'].map(gender_map)

# NEW: Create the same age groups in aml_df to allow the merge
aml_df['age_group'] = pd.cut(aml_df['age_at_diagnosis'], bins=bins, labels=labels)

# 3. Merge on Age Group and Sex (instead of exact age)
final_df = pd.merge(aml_df, lifestyle_map, on=['age_group', 'sex'], how='left')

# Rename columns (Note: use 'age_group' now instead of 'age_at_diagnosis')
lifestyle_map.columns = ['age_group', 'sex', 'predicted_lifestyle_risk']

""" removed - not grouping by exact year anymore
# Group by Age and Sex to get a 'Lifestyle Probability' for those demographics
lifestyle_map = nhanes_df.groupby(['RIDAGEYR', 'RIAGENDR'])['lifestyle_score'].mean().reset_index()

# Rename columns to match for merging
lifestyle_map.columns = ['age_at_diagnosis', 'sex', 'predicted_lifestyle_risk']

"""

# Fill missing values (for ages not in NHANES) with the median risk
median_val = final_df['predicted_lifestyle_risk'].median()
final_df['predicted_lifestyle_risk'] = final_df['predicted_lifestyle_risk'].fillna(median_val)


""" 
# preview data

print("-" * 30)
print(final_df.head())
print("-" * 30)

"""

""" 4. Preparing the Data for Modeling """

# Convert Survival Status to binary (1 for event/death, 0 for censored/alive)
final_df['target_status'] = final_df['overall_survival_status'].apply(lambda x: 1 if 'DECEASED' in str(x).upper() else 0)

# Updated: Categorize Lifestyle Risk
# Instead of using the raw decimal (0.69 correlation with age), 
# we create a "High Risk" vs "Low Risk" group based on the median.
median_risk = final_df['predicted_lifestyle_risk'].median()
final_df['high_lifestyle_risk'] = (final_df['predicted_lifestyle_risk'] > median_risk).astype(int)

# Select features for the model
# We include Age, Lifestyle Risk, and our Retirement proxy
# We use 'high_lifestyle_risk' instead of 'predicted_lifestyle_risk'
model_data = final_df[['age_at_diagnosis', 'high_lifestyle_risk', 'overall_survival_months', 'target_status']]

# Drop any remaining NaNs
model_data = model_data.dropna()

""" 5. Training the Cox Model """

# Initialize and fit the model
cph = CoxPHFitter()
cph.fit(model_data, duration_col='overall_survival_months', event_col='target_status')

# Display the summary table
cph.print_summary()

"""  6. Visualizing the Impact of Retirement 

# Plot survival curves comparing Retired vs. Not Retired
# 1. Create the plot and assign it to a variable 'ax'
ax = cph.plot_partial_effects_on_outcome(
    covariates='is_retired', 
    values=[0, 1], 
    cmap='coolwarm'
)

# 2. Update the legend labels
# The order corresponds to your values list [0, 1]
ax.legend(['Not Retired (0)', 'Retired (1)'])

# 3. Add titles and labels
plt.title("Survival Probability: Working Age (0) vs. Retired (1)")
plt.xlabel("Months since Diagnosis")
plt.ylabel("Survival Probability")
plt.show()

"""

""" 7. The Prediction Logic """
def predict_patient_survival(age, lifestyle_score, model):

    """ old
    # Determine the age group to match the new model training logic
    # This must match the bins used in the merge step
    bins = [0, 30, 45, 60, 75, 120]
    labels = ['Young', 'Adult', 'Mid-Age', 'Senior', 'Elderly']
    age_group = pd.cut([age], bins=bins, labels=labels)[0]
    """
    #updated to match 
    # Logic: If the patient's calculated score is > median, they are High Risk (1)
    # Using 1.0 as a general threshold for a 0-3 scale if median isn't handy, 
    # but ideally use: median_val = final_df['predicted_lifestyle_risk'].median()
    # For now, let's assume if score >= 1, it's High Risk
    is_high_risk = 1 if lifestyle_score >= 1 else 0
    
    # Create a DataFrame for the new patient
    patient_profile = pd.DataFrame({
        'age_at_diagnosis': [age],
        'high_lifestyle_risk': [is_high_risk] # matching the model
    })
    
    # Predict the survival curve
    survival_curve = model.predict_survival_function(patient_profile)
    
    # Extract probabilities for 1-year, 2-year, and 5-year marks
    # Assuming OS_MONTHS is the time unit
    milestones = [12, 24, 60]
    results = {}
    
    for month in milestones:
        # Find the closest time point in the model output
        # 1. Calculate the absolute difference
        idx_diff = (survival_curve.index.to_series() - month).abs()
        
        # 2. Get the integer position of the smallest difference
        closest_pos = idx_diff.argmin()
        
        # 3. Use .iloc with the integer position
        # We use [[closest_pos]] to keep it as a DataFrame/Series
        prob = survival_curve.iloc[[closest_pos]]
        
        # 4. Calculate the percentage
        percentage = round(prob.values[0][0] * 100, 2)
        results[f"{month//12} Year"] = f"{percentage}%"
        
    return results

# Example: A 70-year-old retired patient with a lifestyle risk score of 2.5
# BMI Risk: 1 point if BMI > 30 (obese)
# Smoking Risk: 1 point if smaker
# Sedentary Risk: 1 point if sit more than 8 hours
# lifestyle risk score = BMI pint + smoking point + sedentary point
print("*" * 55)
print("lifestyle risk score = BMI pint + smoking point + sedentary point")
print("BMI Risk: 1 point if BMI > 30 (obese)")
print("Smoking Risk: 1 point if smaker")
print("Sedentary Risk: 1 point if sit more than 8 hours")
print("*" * 55)
print()
print()
print("Example: Predict a 70-year-old retired patient with a lifestyle risk score of 2.5")
# The function now internally converts the 2.5 score into a 1 (High Risk)
prediction = predict_patient_survival(70, 2.5, cph)
print(f"Survival Probabilities: {prediction}")
print()

""" Enter your own """

def calculate_lifestyle_score(bmi, smokes, sedentary_hour):
    """
    Calculate score based on:
    - BMI > 30 (1 point)
    - Smoking (1 = Yes, 2 = No) (1 point)
    - Sedentary > 8 hours (1 point)
    """
    score = 0
    if bmi > 30:
        score += 1
    if smokes == 1:
        score += 1
    if sedentary_hour > 8:
        score += 1
    return score
    
def interactive_prediction(model):
    print("\n--- AML Patient Survival Predictor ---")
    
    # Collect inputs
    try:
        age = int(input("Enter Patient Age: "))
        bmi = float(input("Enter BMI: "))
        smokes = int(input("Did they ever smoke? (1 for yes, 2 for no): "))
        sitting = float(input("Hours spent sitting per day: "))
        
        # 1. Calculate the score
        score = calculate_lifestyle_score(bmi, smokes, sitting)

        # 2. Determine if this is "High Risk" (Binary) to match the model
        # We use 1.0 as the threshold (consistent with predict_patient_survival)
        is_high_risk = 1 if score >= 1 else 0

        # 3. Map the score to a Rank
        ranks = {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}
        risk_label = ranks.get(score, "Unknown")

        
        # 4. Get survival percentages using the existing function
        probs = predict_patient_survival(age, score, model)
        
        # 5. Print Results
        print(f"\nResults for {age}-year-old patient:")
        print(f"Calculated Lifestyle Risk: {score} {risk_label} Risk")
        print(f"Model Classification: {'High Risk' if is_high_risk == 1 else 'Low Risk'}")
        print("-" * 35)
        for year, val in probs.items():
            print(f"{year} Survival Probaility: {val}")
        
        # 6. FIX: Remove is_retired from the plotting profile and Corrected Plotting Logic
        patient_profile = pd.DataFrame({
            'age_at_diagnosis': [age], 
            'high_lifestyle_risk': [is_high_risk]
        })
        
        # 7. Print graph
        patient_curve = model.predict_survival_function(patient_profile)
        
        """ removed old dataframe that include 'is_retired'
        patient_curve = model.predict_survival_function(pd.DataFrame({
            'age_at_diagnosis': [age], 'is_retired': [1 if age >= 65 else 0], 'predicted_lifestyle_risk': [score]
        }))
        """
        
        patient_curve.plot(title=f"Survival Projection: {risk_label} Risk Patient", legend=False)
        plt.ylabel("Probabbility")
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except ValueError:
        print("Invalid input. Please enter numbers only.")
     
    
# RUN (remove # to activate)
#interactive_prediction(cph)


# Select the columns used in your CPH model
cols_to_check = ['age_at_diagnosis', 'high_lifestyle_risk']
corr_matrix = model_data[cols_to_check].corr()

#********************#
#  Plot the heatmap  #
#********************#

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Statistical Validation: Feature Independence")
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig('../visuals/correlation_heatmap.png', dpi=400)

# Display the plot
plt.show()


# 1. Add a constant column (intercept) to the variables
# This is required for a standard 'Centered VIF' calculation
X = add_constant(model_data[cols_to_check].dropna())

# Filter to the numerical columns used in the model
#X = model_data[cols_to_check].dropna()

# Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)


# plot to show two patients-one high risk and one low risk to see survival gap

def plot_risk_comparison(model, avg_age):
    print(f"\nGenerating Comparison Plot for Age {avg_age}...")
    
    # Create two profiles: Low Risk (0) and High Risk (1)
    comparison_df = pd.DataFrame({
        'age_at_diagnosis': [avg_age, avg_age],
        'high_lifestyle_risk': [0, 1]
    }, index=['Low Lifestyle Risk', 'High Lifestyle Risk'])
    
    # Predict survival curves
    curves = model.predict_survival_function(comparison_df)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(curves.index, curves['Low Lifestyle Risk'], color='green', label='Low Risk Lifestyle', linewidth=2)
    plt.plot(curves.index, curves['High Lifestyle Risk'], color='red', label='High Risk Lifestyle', linewidth=2)
    
    plt.title(f"AML Survival Impact: Lifestyle Risk Tiers (Age {avg_age})", fontsize=14)
    plt.xlabel("Months Since Diagnosis", fontsize=12)
    plt.ylabel("Survival Probability", fontsize=12)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    
    # Save a PNG file
    plt.tight_layout()
    plt.savefig('../visuals/survival_comparison_plot.png', dpi=400)
    
    # Display the plot
    plt.show()

# Run the comparison
plot_risk_comparison(cph, avg_age=65)


