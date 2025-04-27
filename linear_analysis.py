import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the CSV file into a DataFrame
merged_df = pd.read_csv('merged.csv')

# Filter out rows where all columns except 'Country' and 'Region' are equal to "Empty"
filtered_df = merged_df.loc[~(merged_df.iloc[:, 2:] == "Empty").all(axis=1)]

filtered_df.to_csv('filtered_df.csv', index=False)
# Remove rows with any cell equal to "Empty"
cleaned_df = filtered_df.loc[~(filtered_df == "Empty").any(axis=1)]

# Convert all columns except 'Country' and 'Region' to numeric
numeric_df = cleaned_df.iloc[:, 2:].apply(pd.to_numeric)

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

# Define the features and target variable
X = numeric_df.drop(columns=['PercentResistant'])
y = numeric_df['PercentResistant']


# Standardize the features
X_standardized = X - X.mean() # Subtract by column mean
X_standardized = X_standardized/X_standardized.std() # Divide by column std

seed = 0
results = pd.DataFrame()
for features, name in zip([X, X_standardized], ["Unnormalized", "Standardized"]):
    res = antibiotic_resistance_lr(X, y, seed)
    res = res.rename(columns ={'Coefficient vals': name})
    results = pd.concat((results, res), axis=1)
    results['Unnormalized * std'] = results['Unnormalized']*X.std()

# Correlations
corrs = numeric_df.corr(method='spearman')
print(corrs)

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

# Correlations
corrs = numeric_df.corr(method='spearman')
#print(corrs)

# Create a new dataframe called dataframe_with_NA where "Empty" is treated as NaN
dataframe_with_NA = merged_df.copy()
dataframe_with_NA.replace("Empty", np.nan, inplace=True)
# Convert all columns except 'Country' and 'Region' to numeric
dataframe_with_NA = dataframe_with_NA.iloc[:, 2:].apply(pd.to_numeric)
print(dataframe_with_NA.columns)