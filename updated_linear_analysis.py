import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the CSV file into a DataFrame
merged_df = pd.read_csv('merged.csv')

# Linear regression function
def antibiotic_resistance_lr(X, y, seed=0): 
    # Split X and y into X_train, X_test and y_train, y_test 
    # # using the same seed ensures that the same rows are picked between X and y 
    X_train = X.sample(frac=0.8, random_state=seed) 
    X_test = X.drop(index=X_train.index) 
    y_train = y.sample(frac=0.8, random_state=seed) 
    y_test = y.drop(index=y_train.index) 

    lr = LinearRegression(fit_intercept=True) 
    # Fit model to data (or train model) 
    lr.fit(X_train, y_train) 
    # Save coefficients of the trained model 
    coefs = pd.DataFrame(lr.coef_, index=lr.feature_names_in_, columns=['Coefficient vals']) 

    # Save model performance on train and test 
    coefs.loc['Train R2 score'] = lr.score(X_train, y_train)
    coefs.loc['Test R2 score'] = lr.score(X_test, y_test) 

    return coefs


# Replace "Empty" with NaN to treat them as missing values
merged_df.replace("Empty", np.nan, inplace=True)

# Drop rows with missing target variable ('PercentResistant')
merged_df.dropna(subset=['PercentResistant'], inplace=True)

# Convert all columns except 'Country' and 'Region' to numeric, ignoring errors
numeric_df = merged_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

# Define the features and target variable
X = numeric_df.drop(columns=['PercentResistant'])
y = numeric_df['PercentResistant']

# Handle missing values in features by filling with column mean
X.fillna(X.mean(), inplace=True)

# Standardize the features
X_standardized = (X - X.mean()) / X.std()

seed = 0
results = pd.DataFrame()
for features, name in zip([X, X_standardized], ["Unnormalized", "Standardized"]):
    res = antibiotic_resistance_lr(features, y, seed)
    res = res.rename(columns={'Coefficient vals': name})
    results = pd.concat((results, res), axis=1)
    results['Unnormalized * std'] = results['Unnormalized'] * X.std()

#results.to_csv('individual_country_results.csv', index=True)
# Calculate and print the R^2 value for the standardized model
standardized_r2 = results.loc['Test R2 score', 'Standardized']
import matplotlib.pyplot as plt

# Ensure y_pred and y_actual have the same size
y_pred = antibiotic_resistance_lr(X_standardized, y, seed).iloc[:, 0]  # Do not slice unless necessary
y_actual = y.iloc[:len(y_pred)]  # Match the size of y_actual to y_pred

# Plot the predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_actual, y_pred, alpha=0.7, label='Predicted vs Actual')
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', label='Ideal Fit')
plt.xlabel('Actual Percent Resistant')
plt.ylabel('Predicted Percent Resistant')
plt.title('Linear Regression: Predicted vs Actual')
plt.legend()
plt.grid(True)
plt.show()

# Print R^2 value
print(f"R^2 value for the standardized model: {standardized_r2}")

# Correlations
corrs = numeric_df.corr(method='spearman')
#print(corrs)
#corrs.to_csv('correlations.csv', index=True)
