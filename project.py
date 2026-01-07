import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1️⃣ LOAD DATA
df = pd.read_csv("diabetes.csv")

print("Original Data Sample:")
print(df.head())


# 2️⃣ REPLACE IMPOSSIBLE ZEROS WITH NaN
cols_with_invalid_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols_with_invalid_zero] = df[cols_with_invalid_zero].replace(0, np.nan)

print("\nMissing values after replacing zeros:")
print(df.isna().sum())


# 3️⃣ IMPUTE MISSING VALUES (MEDIAN)
for col in cols_with_invalid_zero:
    df[col].fillna(df[col].median(), inplace=True)

print("\nMissing values after imputation:")
print(df.isna().sum())


# 4️⃣ REMOVE IMPOSSIBLE VALUES (SAFE CLINICAL LIMITS - EXAMPLES)
df = df[
    (df['Glucose'] >= 40) & (df['Glucose'] <= 400) &
    (df['BloodPressure'] >= 30) & (df['BloodPressure'] <= 200) &
    (df['BMI'] >= 10) & (df['BMI'] <= 70)
]

print("\nShape after removing impossible values:", df.shape)


# 5️⃣ NORMALIZE SELECTED COLUMNS FOR ML
scaler = StandardScaler()
scaled_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']
df_scaled = df.copy()
df_scaled[scaled_cols] = scaler.fit_transform(df[scaled_cols])

print("\nNormalized Sample:")
print(df_scaled.head())


# 6️⃣ SAVE CLEANED DATA
df_scaled.to_csv(r"C:\Users\tanis\OneDrive\Desktop\ML Project\diabetes_cleaned.csv",
                 index=False)

print("\nCLEANED FILE SAVED SUCCESSFULLY 🎯")

###########################################################
#  📊 PLOTS (VISUALIZATION)
###########################################################

# Histogram — Glucose
plt.figure()
plt.hist(df['Glucose'])
plt.title("Glucose Distribution")
plt.xlabel("Glucose")
plt.ylabel("Frequency")
plt.show()

# Histogram — BMI
plt.figure()
plt.hist(df['BMI'])
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()

# Scatter — Age vs Glucose
plt.figure()
plt.scatter(df['Age'], df['Glucose'])
plt.title("Age vs Glucose")
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.show()

# Boxplot — Glucose
plt.figure()
plt.boxplot(df['Glucose'])
plt.title("Glucose Box Plot")
plt.ylabel("Glucose")
plt.show()

# Glucose by Diabetes Outcome
plt.figure()
df.boxplot(column='Glucose', by='Outcome')
plt.title("Glucose by Diabetes Outcome")
plt.suptitle("")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Glucose")
plt.show()

print("\nPLOTS GENERATED SUCCESSFULLY 📊")
